import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowModel(nn.Module):
    def __init__(self, velocity_model: nn.Module):
        super().__init__()
        self.velocity_model = velocity_model

    def flow_matching_loss(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """        
        z: The sample from the dataset
        t: The time step (e.g., in [0,1])

        """
        # sample noise
        eps =  torch.randn_like(z)
        x = t[:, None, None] * z + (1 - t[:, None, None]) * eps
        v = self.velocity_model(x, t)
        loss = F.mse_loss(v, z - eps)
        return loss
    

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        length: int,
        channels: int = 1,
        n_steps: int = 50,
        device: str = "cpu", 
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None
        ) -> torch.Tensor:

        """        
        use Euler method to sample
        """

        self.eval()
        self.velocity_model.eval()

        # can init with a specific sample
        if x0 is None:
            x = torch.randn(batch_size, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h = 1.0 / float(n_steps)
        t = torch.zeros(batch_size, device=device, dtype=dtype)

        for _ in range(n_steps):
            # velocity_model expects t as (B,) and x as (B,C,T)
            u = self.velocity_model(x, t)

            # Euler update: X_{t+h} = X_t + h * u_theta(X_t, t)
            x = x + h * u

            # Update time
            t = t + h

        return x


    def sample_with_guidance(
        self,
        y: torch.Tensor,                          # (B, 1, T) observation
        forward_fn,                               # callable: x -> A(x), differentiable
        batch_size: int,
        length: int,
        channels: int = 1,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        DPS-style guided sampling for rectified-flow / OT-FM.

        Convention (matches flow_matching_loss above):
            x_t = t*z + (1-t)*eps   -- t=0 noise, t=1 data
            v_theta(x_t, t) predicts (z - eps)
            x_hat_0 = x_t + (1-t) * v_theta(x_t, t)

        Posterior step (Bayes: score = unconditional + likelihood):
            v        = v_theta(x_t, t)
            x_hat_0  = x_t + (1-t) * v
            L        = ||y - A(x_hat_0)||^2          # .sum(), per DPS
            g        = grad of L w.r.t. x_t           # flows through v_theta
            x_{t+h}  = x_t + h*v - guidance_scale * g

        NOTE: this method intentionally does NOT use @torch.no_grad() —
        we need autograd through v_theta to compute g. The velocity
        update step runs under torch.no_grad() so the graph does not
        span iterations; each step re-creates a fresh graph via
        x.detach().requires_grad_(True).
        """
        self.eval()
        self.velocity_model.eval()

        if x0 is None:
            x = torch.randn(batch_size, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h = 1.0 / float(n_steps)

        for k in range(n_steps):
            t_val = k * h
            t = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            # fresh graph for this step
            x = x.detach().requires_grad_(True)

            v = self.velocity_model(x, t)
            x_hat_0 = x + (1.0 - t_val) * v                    # data estimate

            residual = y - forward_fn(x_hat_0)
            L = (residual ** 2).sum()                           # DPS: sum, not mean

            grad = torch.autograd.grad(L, x)[0]

            with torch.no_grad():
                x = x + h * v - guidance_scale * grad

        return x.detach()


    def sample_blind_wiener(
        self,
        y: torch.Tensor,                      # (B, 1, T_y)  observation
        batch_size: int,
        length: int,                          # T_x (clap length, model output)
        h_length: int,                        # IR length to estimate
        channels: int = 1,
        n_steps: int = 50,
        guidance_scale: float = 0.01,
        wiener_eps_max: float = 1.0,          # Tikhonov reg at t=0 (heavy — x̂₀ is noise early)
        wiener_eps_min: float = 1e-4,         # Tikhonov reg at t=1 (light — x̂₀ converged)
        likelihood_len: int = None,           # None = full; int = first-N truncation
        h_mode: str = 'wiener',               # 'fixed' | 'ema' | 'wiener'
        h_init: torch.Tensor = None,          # (B, 1, h_length) — required for 'fixed', warm-start for 'ema'/'wiener'
        ema_beta: float = 0.9,                # used only for h_mode='ema'
        tail_max_norm: float = None,          # cap on ||h[1:]||_2; None = no clamp
        peak_norm_after: float = 0.5,         # peak-normalize x once t_val > this
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None,
        ):
        """
        Blind deconv via alternating sampler + closed-form Wiener for h.

        Three h-update modes, each isolating a different diagnostic:

          h_mode='fixed'  : h_est = h_init at EVERY step (no Wiener). Requires
                            h_init. Pre-alternating sanity check — tests whether
                            x-side guidance + peak-norm work in the blind
                            framework with a perfect, frozen IR. If this fails,
                            alternating is moot.

          h_mode='ema'    : k==0 → h_est = h_init; k≥1 →
                            h_est = ema_beta * h_prev + (1 - ema_beta) * h_wiener
                            with the direct-path constraint re-applied after the
                            blend. Tests alternating stability with a smooth,
                            true-IR-anchored h trajectory.

          h_mode='wiener' : pure Wiener every step (k==0 uses h_init if given,
                            else Wiener on noise-x̂₀). This is what M3c will use
                            (cold-start blind).

        Per Euler step k (t_val = k / n_steps):
          (1) h_est ← (per mode above)
          (2) g = grad_x ||y[:L] - (h_est ⊛ x̂₀)[:L]||²
              (autograd through v_θ; h_est is treated as constant)
          (3) x ← x + h*v - guidance_scale * g
              conditionally peak-normalize once t_val > peak_norm_after.

        h_est is PER-SAMPLE (B independent IR estimates of the same room) —
        a built-in consistency diagnostic; for multi-clap shared-IR (variant B),
        the architecture needs to share h across the batch — revisit later.

        Returns (x_recovered, h_estimated), both detached.
        """
        self.eval()
        self.velocity_model.eval()

        assert h_mode in ('fixed', 'ema', 'wiener'), \
            f"unknown h_mode: {h_mode!r} (expected 'fixed' | 'ema' | 'wiener')"
        if h_mode == 'fixed':
            assert h_init is not None, "h_mode='fixed' requires h_init"
        if h_init is not None:
            assert h_init.shape == (batch_size, channels, h_length), (
                f"h_init shape {tuple(h_init.shape)} != "
                f"expected {(batch_size, channels, h_length)}"
            )
            h_init = h_init.to(device=device, dtype=dtype)

        if x0 is None:
            x = torch.randn(batch_size, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h_step = 1.0 / float(n_steps)
        h_est  = None
        h_prev = None       # last-step h_est, used by 'ema'

        for k in range(n_steps):
            t_val = k * h_step
            t = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            x = x.detach().requires_grad_(True)

            v = self.velocity_model(x, t)
            x_hat0 = x + (1.0 - t_val) * v

            # (1) h update — branch on mode
            #
            # Tail-energy clamp order matters: apply AFTER apply_direct_path_
            # constraint (which fixes h[0]=1) so rescale-by-h[0] can't undo
            # the clamp. For EMA, clamp h_wiener BEFORE blending so a single
            # exploding Wiener can't poison the running EMA.
            with torch.no_grad():
                eps_t = wiener_eps_max * (1.0 - t_val) + wiener_eps_min * t_val

                if h_mode == 'fixed':
                    # h_init is trusted; no clamp.
                    h_est = h_init

                elif h_mode == 'ema':
                    if k == 0:
                        h_est = h_init
                    else:
                        h_wiener = wiener_update(x_hat0.detach(), y, h_length, eps_t)
                        # clamp h_wiener BEFORE blending — protects EMA from a
                        # single exploding estimate
                        if tail_max_norm is not None:
                            h_wiener = clamp_tail_energy(h_wiener, tail_max_norm)
                        h_est = ema_beta * h_prev + (1.0 - ema_beta) * h_wiener
                        # both inputs have h[0]=1, so blend has h[0]=1 too —
                        # but re-apply defensively (no-op normally, recovers
                        # from float drift / degeneracy)
                        h_est = apply_direct_path_constraint(h_est)

                elif h_mode == 'wiener':
                    if k == 0 and h_init is not None:
                        h_est = h_init
                    else:
                        h_est = wiener_update(x_hat0.detach(), y, h_length, eps_t)
                        if tail_max_norm is not None:
                            h_est = clamp_tail_energy(h_est, tail_max_norm)

                h_prev = h_est

            # (2) likelihood gradient w.r.t. x, with h_est held constant
            y_pred = grouped_conv_full(x_hat0, h_est)              # (B, 1, T_x + h_length - 1)
            if likelihood_len is not None:
                resid = y[..., :likelihood_len] - y_pred[..., :likelihood_len]
            else:
                resid = y - y_pred
            L = (resid ** 2).sum()
            g = torch.autograd.grad(L, x)[0]

            # (3) guided update + conditional peak-normalize
            with torch.no_grad():
                x = x + h_step * v - guidance_scale * g
                if t_val > peak_norm_after:
                    peak = x.abs().amax(dim=(-1, -2), keepdim=True)
                    x = x / (peak + 1e-8)

        return x.detach(), h_est.detach()  # 2-tuple — M3b convention


    def sample_blind_structured(
        self,
        y: torch.Tensor,                      # (B, 1, T_y)
        batch_size: int,
        length: int,                          # T_x
        h_length: int,
        channels: int = 1,
        n_steps: int = 50,
        guidance_scale: float = 0.01,
        likelihood_len: int = None,
        init_mode: str = 'cold',              # 'fixed' | 'anchored' | 'cold'
        init_w_raw: torch.Tensor = None,      # required for 'fixed' / 'anchored' (from pre-fit)
        init_alpha_raw: torch.Tensor = None,
        init_u: torch.Tensor = None,
        init_w_value: float = 0.4,            # for 'cold' init (softplus output)
        init_alpha_value: float = 7.8125e-4,  # for 'cold' init: α = 1/(80ms@16kHz)
        n_its: int = 10,                      # inner Adam iters per Euler step
        adam_lr: float = 0.1,
        sigma_h_max: float = 1e-2,            # BUDDy gradient-noise reg, t=0 (heavy)
        sigma_h_min: float = 1e-4,            # BUDDy gradient-noise reg, t=1 (light)
        peak_norm_after: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None,
        ):
        """
        BUDDy-style blind deconv: IR is structured as h(n) = δ(n) + w·exp(-α·n)·u(n),
        with (w, α, u) estimated by Adam per Euler step. Inner loop (h estimation)
        and outer step (x guidance) are TWO INDEPENDENT autograd graphs.

        Modes:
          'fixed'    : (w_raw, α_raw, u) pre-fit to true IR (caller's job),
                       n_its forced to 0 — h_est is built once before the loop
                       and reused every step. Isolates the structured-prior
                       ceiling on x-side.
          'anchored' : (w_raw, α_raw, u) pre-fit to true IR, then n_its Adam
                       per step refines them given current x_hat0. Tests
                       whether anchored Adam stays stable.
          'cold'     : (w_raw, α_raw, u) initialized from defaults, n_its Adam
                       per step. True cold-start blind.

        Autograd separation:
          - Inner loop: backward on L_h reaches only (w_raw, α_raw, u). x_hat0
            is detached. velocity_model NEVER appears in the inner graph.
          - Outer step: backward on L_x reaches only x via v_θ. h_est is
            built under no_grad and detached before the outer step.
          - A separation assert is run BEFORE the Euler loop (for 'anchored'
            and 'cold' modes — 'fixed' has no inner loop).

        Returns (x_recovered, h_estimated), both detached.
        """
        import math
        inv_sp = lambda yv: math.log(math.expm1(yv))

        assert init_mode in ('fixed', 'anchored', 'cold'), \
            f"unknown init_mode: {init_mode!r}"
        if init_mode in ('fixed', 'anchored'):
            for tname, t in [('init_w_raw', init_w_raw),
                             ('init_alpha_raw', init_alpha_raw),
                             ('init_u', init_u)]:
                assert t is not None, f"init_mode={init_mode!r} requires {tname}"
            assert init_w_raw.shape     == (batch_size, 1, 1)
            assert init_alpha_raw.shape == (batch_size, 1, 1)
            assert init_u.shape         == (batch_size, channels, h_length)

        self.eval()
        self.velocity_model.eval()

        # ---- initialize learnable IR params ----
        if init_mode == 'cold':
            w_raw = torch.full((batch_size, 1, 1), inv_sp(init_w_value),
                               device=device, dtype=dtype, requires_grad=True)
            alpha_raw = torch.full((batch_size, 1, 1), inv_sp(init_alpha_value),
                                   device=device, dtype=dtype, requires_grad=True)
            u = torch.randn(batch_size, channels, h_length,
                            device=device, dtype=dtype, requires_grad=True)
        else:  # 'fixed' or 'anchored'
            w_raw     = init_w_raw.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
            alpha_raw = init_alpha_raw.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
            u         = init_u.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)

        # ---- persistent Adam across Euler steps (only if inner loop runs) ----
        if init_mode != 'fixed':
            opt_h = torch.optim.Adam([w_raw, alpha_raw, u], lr=adam_lr)

        # ---- two-graph separation assert (only meaningful when inner loop runs) ----
        if init_mode != 'fixed':
            # Build dummy x_hat0 the SAME way the real loop does — through
            # velocity_model, then detach. If the detach is forgotten, the
            # assert below catches it.
            dummy_x = torch.randn(batch_size, channels, length,
                                  device=device, dtype=dtype, requires_grad=True)
            dummy_t = torch.zeros(batch_size, device=device, dtype=dtype)
            dummy_v = self.velocity_model(dummy_x, dummy_t)
            dummy_x_hat0 = dummy_x + dummy_v          # has grad path to velocity_model
            dummy_x_det  = dummy_x_hat0.detach()      # if this detach is missing, assert fires

            h_test = build_structured_ir(F.softplus(w_raw), F.softplus(alpha_raw),
                                         u, h_length, device)
            y_test = grouped_conv_full_diffh(dummy_x_det, h_test)
            L_test = ((y - y_test) ** 2).sum()

            sample_param = next(self.velocity_model.parameters())
            saved_grad   = sample_param.grad
            sample_param.grad = None
            # clear inner-leaf grads too (Adam.zero_grad equivalent)
            for p in (w_raw, alpha_raw, u):
                if p.grad is not None:
                    p.grad = None

            L_test.backward()

            assert sample_param.grad is None, (
                "two-graph separation broken: inner-loop backward reached a "
                "velocity_model parameter. Check that x_hat0.detach() is "
                "applied before the inner loop, and that grouped_conv_full_diffh "
                "operates on the detached x_det only."
            )
            sample_param.grad = saved_grad
            # leaf grads will be cleared again by opt_h.zero_grad() at first inner iter

        # ---- 'fixed' mode: build h_est once and reuse ----
        if init_mode == 'fixed':
            with torch.no_grad():
                h_frozen = build_structured_ir(
                    F.softplus(w_raw), F.softplus(alpha_raw), u, h_length, device
                ).detach()

        # ---- initialize x trajectory ----
        if x0 is None:
            x = torch.randn(batch_size, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h_step = 1.0 / float(n_steps)
        h_est  = None

        # ---- main Euler loop ----
        for k in range(n_steps):
            t_val = k * h_step
            t = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            x = x.detach().requires_grad_(True)
            v = self.velocity_model(x, t)
            x_hat0 = x + (1.0 - t_val) * v

            # (1) INNER LOOP — estimate (w_raw, α_raw, u). Skipped if 'fixed'.
            if init_mode == 'fixed':
                h_est = h_frozen
            else:
                x_det = x_hat0.detach()
                sigma_h = sigma_h_max * (1.0 - t_val) + sigma_h_min * t_val
                for _ in range(n_its):
                    w_pos     = F.softplus(w_raw)
                    alpha_pos = F.softplus(alpha_raw)
                    h_param   = build_structured_ir(w_pos, alpha_pos, u, h_length, device)
                    y_pred_h  = grouped_conv_full_diffh(x_det, h_param)
                    if likelihood_len is not None:
                        resid_h = y[..., :likelihood_len] - y_pred_h[..., :likelihood_len]
                    else:
                        resid_h = y - y_pred_h
                    L_h_data = (resid_h ** 2).sum()
                    # BUDDy eq.11 annealed gradient-noise injection. The VALUE
                    # of L_reg is irrelevant; its GRADIENT w.r.t. h_param is
                    # σ_h · noise. Do NOT "simplify" this away.
                    L_reg = sigma_h * (h_param * torch.randn_like(h_param)).sum()
                    L_total = L_h_data + L_reg
                    opt_h.zero_grad()
                    L_total.backward()
                    opt_h.step()

                # build the frozen estimate for the outer x-gradient
                with torch.no_grad():
                    h_est = build_structured_ir(
                        F.softplus(w_raw), F.softplus(alpha_raw), u, h_length, device
                    ).detach()

            # (2) OUTER STEP — x guidance, h_est held constant
            y_pred = grouped_conv_full(x_hat0, h_est)
            if likelihood_len is not None:
                resid = y[..., :likelihood_len] - y_pred[..., :likelihood_len]
            else:
                resid = y - y_pred
            L_x = (resid ** 2).sum()
            g = torch.autograd.grad(L_x, x)[0]

            with torch.no_grad():
                x = x + h_step * v - guidance_scale * g
                if t_val > peak_norm_after:
                    peak = x.abs().amax(dim=(-1, -2), keepdim=True)
                    x = x / (peak + 1e-8)

        # 4-tuple: caller can inspect the final (w, α) envelope vs the pre-fit
        # reference to see whether Adam converged to the right structured params.
        with torch.no_grad():
            w_final     = F.softplus(w_raw).detach()
            alpha_final = F.softplus(alpha_raw).detach()
        return x.detach(), h_est.detach(), w_final, alpha_final


    def sample_blind_structured_shared(
        self,
        y: torch.Tensor,                      # (N, 1, T_y)  N observations
        batch_size: int,                      # N (number of claps sharing one IR)
        length: int,
        h_length: int,
        channels: int = 1,
        n_steps: int = 50,
        guidance_scale: float = 0.01,
        likelihood_len: int = None,
        init_mode: str = 'cold',
        init_w_raw: torch.Tensor = None,      # (1,1,1) — SHARED, not (N,...)
        init_alpha_raw: torch.Tensor = None,  # (1,1,1)
        init_u: torch.Tensor = None,          # (1, channels, h_length)
        init_w_value: float = 0.4,
        init_alpha_value: float = 7.8125e-4,
        n_its: int = 10,
        adam_lr: float = 0.1,
        sigma_h_max: float = 1e-2,
        sigma_h_min: float = 1e-4,
        peak_norm_after: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None,
        debug_grad: bool = False,             # INSTRUMENTATION ONLY — see inner loop
        use_oracle_x: bool = False,           # DIAGNOSTIC — see inner loop
        x_true: torch.Tensor = None,          # (N, channels, T_x) — required if use_oracle_x
        ):
        """
        Variant-B step 1 (synthetic batch form): multi-clap blind deconv with
        a SHARED structured IR. N different claps x_i, all convolved with the
        SAME h. Estimating one shared (w_raw, α_raw, u) — gradient aggregates
        across the N likelihood terms in a single backward.

        Shape contract (the differences from sample_blind_structured):
          x trajectory : (N, 1, T_x)        — N independent chains, unchanged
          y observation: (N, 1, T_y)        — one per clap
          (w_raw, α_raw): (1, 1, 1)         — SHARED scalar pair
          u            : (1, channels, h_length)   — SHARED tail
          h_est        : (1, 1, h_length)   — single IR, returned as-is

        The N-fold constraint comes from L_h_data = sum over (N × T_y) terms;
        h_param appears in all N terms via grouped_conv_full_sharedh; one
        backward aggregates the gradient onto the single (w_raw, α_raw, u).

        x-side is per-sample, identical to B2-time: each x_i gets its own
        DPS gradient through its own v_θ. Only h is shared.

        Form (b) — concatenate (real data): one long signal [clap1, gap,
        clap2, ...] ⊛ one IR — uses the SAME shared-h estimation logic; only
        the y/x layout differs. Add a separate entry point or preprocessing
        wrapper when real-data scripts need it. Not implemented here.

        Returns (x_recovered (N,1,T_x), h_estimated (1,1,h_length),
                 w_final (1,1,1), alpha_final (1,1,1)).
        """
        import math
        inv_sp = lambda yv: math.log(math.expm1(yv))

        N = batch_size

        assert init_mode in ('fixed', 'anchored', 'cold'), \
            f"unknown init_mode: {init_mode!r}"
        if init_mode in ('fixed', 'anchored'):
            for tname, t in [('init_w_raw', init_w_raw),
                             ('init_alpha_raw', init_alpha_raw),
                             ('init_u', init_u)]:
                assert t is not None, f"init_mode={init_mode!r} requires {tname}"
            assert init_w_raw.shape     == (1, 1, 1), \
                f"shared init_w_raw shape must be (1,1,1), got {tuple(init_w_raw.shape)}"
            assert init_alpha_raw.shape == (1, 1, 1), \
                f"shared init_alpha_raw shape must be (1,1,1), got {tuple(init_alpha_raw.shape)}"
            assert init_u.shape         == (1, channels, h_length), \
                f"shared init_u shape must be (1,{channels},{h_length}), got {tuple(init_u.shape)}"

        if use_oracle_x:
            assert x_true is not None, (
                "use_oracle_x=True requires x_true (the true claps to feed "
                "the inner h-estimation in place of the running estimate)"
            )
            assert x_true.shape == (batch_size, channels, length), (
                f"x_true shape {tuple(x_true.shape)} != "
                f"expected {(batch_size, channels, length)}"
            )

        self.eval()
        self.velocity_model.eval()

        # ---- SHARED IR params: batch dim 1 (not N) ----
        if init_mode == 'cold':
            w_raw     = torch.full((1, 1, 1), inv_sp(init_w_value),
                                   device=device, dtype=dtype, requires_grad=True)
            alpha_raw = torch.full((1, 1, 1), inv_sp(init_alpha_value),
                                   device=device, dtype=dtype, requires_grad=True)
            u         = torch.randn(1, channels, h_length,
                                    device=device, dtype=dtype, requires_grad=True)
        else:
            w_raw     = init_w_raw.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
            alpha_raw = init_alpha_raw.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
            u         = init_u.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)

        if init_mode != 'fixed':
            opt_h = torch.optim.Adam([w_raw, alpha_raw, u], lr=adam_lr)

        # ---- two-graph separation assert (carried over from B2-time) ----
        if init_mode != 'fixed':
            dummy_x = torch.randn(N, channels, length,
                                  device=device, dtype=dtype, requires_grad=True)
            dummy_t = torch.zeros(N, device=device, dtype=dtype)
            dummy_v = self.velocity_model(dummy_x, dummy_t)
            dummy_x_hat0 = dummy_x + dummy_v
            dummy_x_det  = dummy_x_hat0.detach()

            h_test = build_structured_ir(F.softplus(w_raw), F.softplus(alpha_raw),
                                         u, h_length, device)       # (1,1,h_length)
            assert h_test.shape == (1, 1, h_length), (
                f"build_structured_ir output shape {tuple(h_test.shape)} != "
                f"(1,1,{h_length}); shared-h is broken"
            )
            y_test = grouped_conv_full_sharedh(dummy_x_det, h_test)   # (N,1,T_y)
            L_test = ((y - y_test) ** 2).sum()

            sample_param = next(self.velocity_model.parameters())
            saved_grad   = sample_param.grad
            sample_param.grad = None
            for p in (w_raw, alpha_raw, u):
                if p.grad is not None:
                    p.grad = None

            L_test.backward()

            assert sample_param.grad is None, (
                "two-graph separation broken: inner-loop backward reached "
                "velocity_model parameter (shared-h variant)"
            )
            # also verify the shared-h gradient is non-trivial — if it were
            # zero, the N-fold aggregation isn't happening
            assert w_raw.grad is not None and u.grad is not None, (
                "shared-h gradient did not reach (w_raw, u) — check that "
                "grouped_conv_full_sharedh reuses the same h_shared tensor"
            )
            sample_param.grad = saved_grad

        # ---- 'fixed': build shared h_est once ----
        if init_mode == 'fixed':
            with torch.no_grad():
                h_frozen = build_structured_ir(
                    F.softplus(w_raw), F.softplus(alpha_raw), u, h_length, device
                ).detach()                                            # (1,1,h_length)

        # ---- initialize x trajectory (N independent chains) ----
        if x0 is None:
            x = torch.randn(N, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h_step = 1.0 / float(n_steps)
        h_est  = None

        for k in range(n_steps):
            t_val = k * h_step
            t = torch.full((N,), t_val, device=device, dtype=dtype)

            x = x.detach().requires_grad_(True)
            v = self.velocity_model(x, t)
            x_hat0 = x + (1.0 - t_val) * v                            # (N,1,T_x)

            # (1) INNER LOOP — shared (w_raw, α_raw, u); skip if 'fixed'
            if init_mode == 'fixed':
                h_est = h_frozen                                       # (1,1,h_length)
            else:
                # x_det is the constant x the inner Adam sees for h-estimation.
                # Oracle mode swaps it for the true claps; the OUTER x-guidance
                # below is UNCHANGED and still uses x_hat0 (the real running
                # estimate). So oracle isolates "if x were perfect, can the
                # N-clap likelihood identify h?" without making the whole
                # problem non-blind.
                if use_oracle_x:
                    x_det = x_true.detach()
                else:
                    x_det = x_hat0.detach()
                sigma_h = sigma_h_max * (1.0 - t_val) + sigma_h_min * t_val
                for inner_it in range(n_its):
                    w_pos     = F.softplus(w_raw)
                    alpha_pos = F.softplus(alpha_raw)
                    h_param   = build_structured_ir(w_pos, alpha_pos, u,
                                                    h_length, device)  # (1,1,h_length)
                    y_pred_h  = grouped_conv_full_sharedh(x_det, h_param)   # (N,1,T_y)
                    if likelihood_len is not None:
                        resid_h = y[..., :likelihood_len] - y_pred_h[..., :likelihood_len]
                    else:
                        resid_h = y - y_pred_h
                    # sums over N AND time — N-fold constraint on the shared IR.
                    # Same h_param appears in all N residual terms; one backward
                    # aggregates the gradient onto (w_raw, α_raw, u).
                    L_h_data = (resid_h ** 2).sum()
                    # BUDDy eq.11 annealed gradient-noise injection — value
                    # irrelevant, gradient = σ_h · noise. Do NOT simplify away.
                    L_reg = sigma_h * (h_param * torch.randn_like(h_param)).sum()
                    L_total = L_h_data + L_reg
                    opt_h.zero_grad()
                    L_total.backward()

                    # ---- DIAGNOSTIC (instrumentation only) ----------------
                    # Fires exactly ONCE per run (first Euler step, first
                    # inner iter) when debug_grad=True. Verifies the
                    # gradient on the SHARED (w_raw, α_raw, u) truly
                    # aggregates contributions from all N claps, not just
                    # clap 0. Save/restore makes opt_h.step() see the
                    # original aggregated grad — the real optimization
                    # trajectory is byte-for-byte unchanged.
                    if debug_grad and k == 0 and inner_it == 0:
                        print(f"  [diagnostic] N={N}, k=0, inner_it=0")
                        # Check 1: aggregated grad norms (already on the leaves)
                        print(f"    aggregated grad norms (the real gradient that "
                              f"opt_h.step() will use):")
                        print(f"      w_raw.grad.norm()     = "
                              f"{w_raw.grad.norm().item():.6e}")
                        print(f"      alpha_raw.grad.norm() = "
                              f"{alpha_raw.grad.norm().item():.6e}")
                        print(f"      u.grad.norm()         = "
                              f"{u.grad.norm().item():.6e}")

                        # Save aggregated grads BEFORE Check 2 corrupts them.
                        # .detach().clone() decouples from any future ops.
                        saved_w     = w_raw.grad.detach().clone()
                        saved_alpha = alpha_raw.grad.detach().clone()
                        saved_u     = u.grad.detach().clone()

                        # Check 2: per-clap contribution decomposition.
                        # For each i, zero grads, build h_param fresh, run
                        # backward on clap-i-only residual, print grad norms.
                        # h_param is rebuilt each time because the original
                        # graph was freed by .backward() above.
                        print(f"    per-clap contribution decomposition:")
                        for i in range(N):
                            w_raw.grad     = None
                            alpha_raw.grad = None
                            u.grad         = None
                            h_param_i = build_structured_ir(
                                F.softplus(w_raw), F.softplus(alpha_raw),
                                u, h_length, device
                            )
                            y_pred_i = grouped_conv_full_sharedh(
                                x_det[i:i+1], h_param_i
                            )                                     # (1,1,T_y)
                            if likelihood_len is not None:
                                resid_i = (y[i:i+1, ..., :likelihood_len]
                                           - y_pred_i[..., :likelihood_len])
                            else:
                                resid_i = y[i:i+1] - y_pred_i
                            L_i = (resid_i ** 2).sum()
                            L_i.backward()
                            print(f"      clap {i+1}/{N}: "
                                  f"w_raw.grad.norm()={w_raw.grad.norm().item():.6e}, "
                                  f"alpha_raw.grad.norm()={alpha_raw.grad.norm().item():.6e}, "
                                  f"u.grad.norm()={u.grad.norm().item():.6e}")

                        # Restore aggregated grads so opt_h.step() uses the
                        # CORRECT (pre-diagnostic) gradient. Without this,
                        # opt_h.step() would step using only clap N-1's grad
                        # from the last decomposition iteration.
                        w_raw.grad     = saved_w
                        alpha_raw.grad = saved_alpha
                        u.grad         = saved_u
                        print(f"    aggregated grads restored — real Adam step "
                              f"proceeds with the original gradient")
                    # ---- END DIAGNOSTIC ------------------------------------

                    opt_h.step()

                with torch.no_grad():
                    h_est = build_structured_ir(
                        F.softplus(w_raw), F.softplus(alpha_raw), u, h_length, device
                    ).detach()                                         # (1,1,h_length)

            # (2) OUTER STEP — x guidance, shared h held constant
            y_pred = grouped_conv_full_sharedh(x_hat0, h_est)          # (N,1,T_y)
            if likelihood_len is not None:
                resid = y[..., :likelihood_len] - y_pred[..., :likelihood_len]
            else:
                resid = y - y_pred
            L_x = (resid ** 2).sum()
            g = torch.autograd.grad(L_x, x)[0]                        # (N,1,T_x)

            with torch.no_grad():
                x = x + h_step * v - guidance_scale * g
                if t_val > peak_norm_after:
                    peak = x.abs().amax(dim=(-1, -2), keepdim=True)
                    x = x / (peak + 1e-8)

        with torch.no_grad():
            w_final     = F.softplus(w_raw).detach()                  # (1,1,1)
            alpha_final = F.softplus(alpha_raw).detach()              # (1,1,1)
        return x.detach(), h_est.detach(), w_final, alpha_final


    def sample_blind_stft_structured(
        self,
        y: torch.Tensor,                      # (B, 1, T_y)
        batch_size: int,
        length: int,                          # T_x
        h_length: int,
        channels: int = 1,
        n_steps: int = 50,
        xi_x: float = 1.0,                    # BUDDy gradient-norm-normalized DPS step size
        n_its: int = 5,                       # inner Adam iters per Euler step
        adam_lr: float = 0.01,
        c_compress: float = 0.5,              # BUDDy magnitude compression exponent
        stft_win: int = 1024,
        stft_hop: int = 256,
        init_mode: str = 'cold',              # 'fixed' | 'cold'
        fixed_h: torch.Tensor = None,         # (B,1,h_length) — required for 'fixed' (the true IR)
        init_w_value: float = 0.1,
        init_alpha_value: float = 0.2,        # PER-FRAME decay (≈ 7.8e-4/sample × hop). NOT the
                                              # time-domain 7.8e-4 — α here multiplies frame index.
        init_phi_scale: float = 1e-3,
        sigma_h_max: float = 1e-2,            # BUDDy gradient-noise reg, t=0 (heavy)
        sigma_h_min: float = 1e-4,            # BUDDy gradient-noise reg, t=1 (light)
        peak_norm_after: float = 0.8,
        use_oracle_x: bool = False,           # DIAGNOSTIC — inner h-est sees true claps
        x_true: torch.Tensor = None,          # (B, channels, T_x) — required if use_oracle_x
        phase_unfreeze_after: float = 0.0,    # phase-freedom annealing: freeze phi for t_val < threshold
        debug_phase: bool = False,            # DIAGNOSTIC — print grad/phase stats; real path unchanged
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None,
        ):
        """
        B2-stft: STFT-domain analog of sample_blind_structured. The IR is
        parameterized in the STFT domain — structured per-bin magnitude
        envelope |H[k,n]| = w[k]·exp(-α[k]·n) plus a FREE per-bin phase
        φ[k,n] — instead of the time-domain noise vector u. Free phase is the
        parameter the time-domain u could not represent (env_corr(h) capped at
        0.43, §5c). (w, α, φ) estimated by Adam per Euler step.

        Key differences from B2-time:
          - parameterization: STFT (w[F], α[F], φ[F,T_frames]) vs time (w,α scalars, u[h_length])
          - likelihood: magnitude-compressed STFT domain (BUDDy, c=0.5) vs raw time-domain MSE
          - DPS x-step: gradient-norm normalized (xi_x·g/‖g‖, BUDDy) vs guidance_scale·g

        Modes:
          'fixed' : h = fixed_h (the TRUE IR), frozen, no inner loop. Sanity-
                    checks that the STFT x-side machinery works given perfect h.
          'cold'  : (w,α,φ) cold-init, n_its Adam per step. True blind.
          use_oracle_x=True (with 'cold'): inner h-est sees x_true; the OUTER
                    x-guidance still uses the running x_hat0, so x_recovered is
                    a genuine blind result — isolates "is φ identifiable when x
                    is given?" (the §5c oracle-x test, in the STFT domain).

        Autograd separation (mirrors B2-time): inner backward reaches only
        (w,α,φ); outer reaches only x. Verified by an explicit assert.

        peak-norm is applied AFTER the DPS step under no_grad — never inside the
        likelihood gradient (backprop must not pass through it).

        Returns (x_recovered, h_estimated, w_final (B,F), α_final (B,F),
                 φ_final (B,F,T_frames)), all detached.
        """
        import math
        inv_sp = lambda yv: math.log(math.expm1(yv))

        assert init_mode in ('fixed', 'cold'), f"unknown init_mode: {init_mode!r}"
        if init_mode == 'fixed':
            assert fixed_h is not None, "init_mode='fixed' requires fixed_h (the true IR)"
            assert fixed_h.shape == (batch_size, 1, h_length), \
                f"fixed_h shape {tuple(fixed_h.shape)} != {(batch_size,1,h_length)}"
        if use_oracle_x:
            assert x_true is not None, "use_oracle_x=True requires x_true"
            assert x_true.shape == (batch_size, channels, length), \
                f"x_true shape {tuple(x_true.shape)} != {(batch_size,channels,length)}"

        self.eval()
        self.velocity_model.eval()

        # ---- STFT grid for the IR parameterization ----
        window = torch.hann_window(stft_win, device=device, dtype=dtype)
        Fbins  = stft_win // 2 + 1
        with torch.no_grad():
            dummy = torch.zeros(1, h_length, device=device, dtype=dtype)
            T_frames = torch.stft(dummy, n_fft=stft_win, hop_length=stft_hop,
                                  win_length=stft_win, window=window, center=True,
                                  return_complex=True, onesided=True).shape[-1]
        frame_idx = torch.arange(T_frames, device=device, dtype=dtype)

        # ---- initialize learnable STFT-IR params ----
        w_raw     = torch.full((batch_size, Fbins), inv_sp(init_w_value),
                               device=device, dtype=dtype, requires_grad=True)
        alpha_raw = torch.full((batch_size, Fbins), inv_sp(init_alpha_value),
                               device=device, dtype=dtype, requires_grad=True)
        phi       = (init_phi_scale * torch.randn(batch_size, Fbins, T_frames,
                     device=device, dtype=dtype)).detach().requires_grad_(True)

        if init_mode != 'fixed':
            opt_h = torch.optim.Adam([w_raw, alpha_raw, phi], lr=adam_lr)

        def _build_h():
            return build_stft_ir(F.softplus(w_raw), F.softplus(alpha_raw), phi,
                                 h_length, stft_win, stft_hop, window, frame_idx)

        # ---- two-graph separation assert (only when inner loop runs) ----
        if init_mode != 'fixed':
            dummy_x = torch.randn(batch_size, channels, length,
                                  device=device, dtype=dtype, requires_grad=True)
            dummy_t = torch.zeros(batch_size, device=device, dtype=dtype)
            dummy_v = self.velocity_model(dummy_x, dummy_t)
            dummy_x_hat0 = dummy_x + dummy_v
            dummy_x_det  = dummy_x_hat0.detach()

            h_test = _build_h()
            assert h_test.shape == (batch_size, 1, h_length), \
                f"build_stft_ir output {tuple(h_test.shape)} != {(batch_size,1,h_length)}"
            y_test = grouped_conv_full_diffh(dummy_x_det, h_test)
            L_test = stft_mag_compress_loss(y, y_test, c_compress,
                                            stft_win, stft_hop, window)

            sample_param = next(self.velocity_model.parameters())
            saved_grad   = sample_param.grad
            sample_param.grad = None
            for p in (w_raw, alpha_raw, phi):
                if p.grad is not None:
                    p.grad = None

            L_test.backward()

            assert sample_param.grad is None, (
                "two-graph separation broken: STFT inner-loop backward reached a "
                "velocity_model parameter — check x_hat0.detach() before the inner loop"
            )
            assert w_raw.grad is not None and phi.grad is not None, (
                "STFT inner-loop gradient did not reach (w_raw, φ) — check the "
                "build_stft_ir → conv → compressed-likelihood pipeline"
            )
            sample_param.grad = saved_grad

        # ---- 'fixed' mode: use the true IR directly ----
        if init_mode == 'fixed':
            h_frozen = fixed_h.detach().to(device=device, dtype=dtype)

        # ---- initialize x trajectory ----
        if x0 is None:
            x = torch.randn(batch_size, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h_step = 1.0 / float(n_steps)
        h_est  = None

        for k in range(n_steps):
            t_val = k * h_step
            t = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            x = x.detach().requires_grad_(True)
            v = self.velocity_model(x, t)
            x_hat0 = x + (1.0 - t_val) * v

            # Eloi fix: normalize the Tweedie estimate x_hat0 (NOT x) so the
            # operator and gradient see a stable amplitude scale, while leaving
            # the noisy x and its implicit noise level untouched.
            with torch.no_grad():
                peak_hat = x_hat0.abs().amax(dim=(-1, -2), keepdim=True)
            x_hat0 = x_hat0 / (peak_hat + 1e-8)   # gradient still flows x→v→x_hat0

            # (1) INNER LOOP — estimate (w, α, φ); skipped if 'fixed'
            if init_mode == 'fixed':
                h_est = h_frozen
            else:
                x_det = x_true.detach() if use_oracle_x else x_hat0.detach()
                sigma_h = sigma_h_max * (1.0 - t_val) + sigma_h_min * t_val
                for _inner_it in range(n_its):
                    h_param  = _build_h()
                    y_pred_h = grouped_conv_full_diffh(x_det, h_param)
                    L_data   = stft_mag_compress_loss(y, y_pred_h, c_compress,
                                                      stft_win, stft_hop, window)
                    # BUDDy eq.11 annealed gradient-noise injection — value
                    # irrelevant, gradient = σ_h · noise. Do NOT simplify away.
                    L_reg    = sigma_h * (h_param * torch.randn_like(h_param)).sum()
                    L_total  = L_data + L_reg
                    opt_h.zero_grad()
                    L_total.backward()
                    # CHECK 1: is phi receiving gradient? (k=0, inner=0 only)
                    if debug_phase and k == 0 and _inner_it == 0:
                        print(f"  [CHECK 1] k=0 inner=0 grad norms: "
                              f"phi={phi.grad.norm().item():.4e}  "
                              f"w_raw={w_raw.grad.norm().item():.4e}  "
                              f"alpha_raw={alpha_raw.grad.norm().item():.4e}")
                    # phase-freedom annealing: freeze phi until t_val >= threshold
                    if t_val < phase_unfreeze_after and phi.grad is not None:
                        phi.grad.zero_()
                    opt_h.step()

                with torch.no_grad():
                    h_est = _build_h().detach()

            # (2) OUTER STEP — x guidance, h_est held constant, gradient-norm
            #     normalized DPS step (BUDDy). Compression-domain likelihood.
            y_pred = grouped_conv_full(x_hat0, h_est)
            L_x = stft_mag_compress_loss(y, y_pred, c_compress,
                                         stft_win, stft_hop, window)
            g = torch.autograd.grad(L_x, x)[0]                         # (B,1,T_x)

            with torch.no_grad():
                # per-sample gradient-norm normalization: each clap's DPS step
                # has unit-normalized direction scaled by xi_x (robust to the
                # compression's per-sample magnitude swings — Eloi's BUDDy tip).
                g_norm = g.flatten(1).norm(dim=1).view(-1, 1, 1)       # (B,1,1)
                g_dir  = g / (g_norm + 1e-8)
                x = x + h_step * v - xi_x * g_dir
                # peak_norm on x REMOVED — normalizing x shifts the noise level
                # so t no longer matches training (Eloi, Jun 2026). x_hat0 is
                # normalized above before the inner loop instead.

        with torch.no_grad():
            w_final     = F.softplus(w_raw).detach()                  # (B, F)
            alpha_final = F.softplus(alpha_raw).detach()              # (B, F)
            phi_final   = phi.detach()                                # (B, F, T_frames)
        return x.detach(), h_est.detach(), w_final, alpha_final, phi_final


    def sample_blind_stft_structured_shared(
        self,
        y: torch.Tensor,                      # (N, 1, T_y)  N observations
        batch_size: int,                      # N (claps sharing one IR)
        length: int,
        h_length: int,
        channels: int = 1,
        n_steps: int = 50,
        xi_x: float = 1.0,
        n_its: int = 5,
        adam_lr: float = 0.01,
        c_compress: float = 0.5,
        stft_win: int = 1024,
        stft_hop: int = 256,
        init_mode: str = 'cold',              # 'fixed' | 'cold'
        fixed_h: torch.Tensor = None,         # (1,1,h_length) — SHARED true IR for 'fixed'
        init_w_value: float = 0.1,
        init_alpha_value: float = 0.2,
        init_phi_scale: float = 1e-3,
        sigma_h_max: float = 1e-2,
        sigma_h_min: float = 1e-4,
        peak_norm_after: float = 0.8,
        use_oracle_x: bool = False,
        x_true: torch.Tensor = None,          # (N, channels, T_x) — required if use_oracle_x
        phase_unfreeze_after: float = 0.0,    # phase-freedom annealing: freeze phi for t_val < threshold
        debug_grad: bool = False,             # INSTRUMENTATION ONLY (per-clap grad decomposition)
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x0: torch.Tensor = None,
        ):
        """
        Variant-B + B2-stft: multi-clap blind deconv with a SHARED STFT-domain
        structured IR. N different claps x_i, all convolved with the SAME h.
        ONE shared (w, α, φ); the inner loss sums over the N likelihood terms
        so one backward aggregates the N-fold constraint onto the shared params.

        This is the final configuration from §5c: free per-bin phase φ is what
        the N-clap complementary constraints can pin down (the time-domain
        shared u could not — env_corr(h) was flat in N). Expect env_corr(h) to
        CLIMB with N here.

        Shape contract (differences from the single form):
          x trajectory : (N, 1, T_x)            N independent chains
          (w, α)       : (1, F)                 SHARED
          φ            : (1, F, T_frames)        SHARED
          h_est        : (1, 1, h_length)        single IR (grouped_conv_full_sharedh)

        x-side is per-sample (each x_i its own DPS gradient, per-sample
        gradient-norm normalized). Only h is shared.

        Returns (x_recovered (N,1,T_x), h_estimated (1,1,h_length),
                 w_final (1,F), α_final (1,F), φ_final (1,F,T_frames)).
        """
        import math
        inv_sp = lambda yv: math.log(math.expm1(yv))
        N = batch_size

        assert init_mode in ('fixed', 'cold'), f"unknown init_mode: {init_mode!r}"
        if init_mode == 'fixed':
            assert fixed_h is not None, "init_mode='fixed' requires fixed_h"
            assert fixed_h.shape == (1, 1, h_length), \
                f"shared fixed_h shape {tuple(fixed_h.shape)} != (1,1,{h_length})"
        if use_oracle_x:
            assert x_true is not None, "use_oracle_x=True requires x_true"
            assert x_true.shape == (N, channels, length), \
                f"x_true shape {tuple(x_true.shape)} != {(N,channels,length)}"

        self.eval()
        self.velocity_model.eval()

        window = torch.hann_window(stft_win, device=device, dtype=dtype)
        Fbins  = stft_win // 2 + 1
        with torch.no_grad():
            dummy = torch.zeros(1, h_length, device=device, dtype=dtype)
            T_frames = torch.stft(dummy, n_fft=stft_win, hop_length=stft_hop,
                                  win_length=stft_win, window=window, center=True,
                                  return_complex=True, onesided=True).shape[-1]
        frame_idx = torch.arange(T_frames, device=device, dtype=dtype)

        # ---- SHARED STFT-IR params: batch dim 1 (not N) ----
        w_raw     = torch.full((1, Fbins), inv_sp(init_w_value),
                               device=device, dtype=dtype, requires_grad=True)
        alpha_raw = torch.full((1, Fbins), inv_sp(init_alpha_value),
                               device=device, dtype=dtype, requires_grad=True)
        phi       = (init_phi_scale * torch.randn(1, Fbins, T_frames,
                     device=device, dtype=dtype)).detach().requires_grad_(True)

        if init_mode != 'fixed':
            opt_h = torch.optim.Adam([w_raw, alpha_raw, phi], lr=adam_lr)

        def _build_h():
            return build_stft_ir(F.softplus(w_raw), F.softplus(alpha_raw), phi,
                                 h_length, stft_win, stft_hop, window, frame_idx)

        # ---- two-graph separation assert ----
        if init_mode != 'fixed':
            dummy_x = torch.randn(N, channels, length,
                                  device=device, dtype=dtype, requires_grad=True)
            dummy_t = torch.zeros(N, device=device, dtype=dtype)
            dummy_v = self.velocity_model(dummy_x, dummy_t)
            dummy_x_det = (dummy_x + dummy_v).detach()

            h_test = _build_h()                                        # (1,1,h_length)
            assert h_test.shape == (1, 1, h_length), \
                f"shared build_stft_ir output {tuple(h_test.shape)} != (1,1,{h_length})"
            y_test = grouped_conv_full_sharedh(dummy_x_det, h_test)    # (N,1,T_y)
            L_test = stft_mag_compress_loss(y, y_test, c_compress,
                                            stft_win, stft_hop, window)

            sample_param = next(self.velocity_model.parameters())
            saved_grad   = sample_param.grad
            sample_param.grad = None
            for p in (w_raw, alpha_raw, phi):
                if p.grad is not None:
                    p.grad = None

            L_test.backward()

            assert sample_param.grad is None, (
                "two-graph separation broken: STFT shared inner-loop backward "
                "reached a velocity_model parameter"
            )
            assert w_raw.grad is not None and phi.grad is not None, (
                "shared STFT gradient did not reach (w_raw, φ) — check "
                "grouped_conv_full_sharedh reuses the same h tensor"
            )
            sample_param.grad = saved_grad

        if init_mode == 'fixed':
            h_frozen = fixed_h.detach().to(device=device, dtype=dtype)

        if x0 is None:
            x = torch.randn(N, channels, length, device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)

        h_step = 1.0 / float(n_steps)
        h_est  = None

        for k in range(n_steps):
            t_val = k * h_step
            t = torch.full((N,), t_val, device=device, dtype=dtype)

            x = x.detach().requires_grad_(True)
            v = self.velocity_model(x, t)
            x_hat0 = x + (1.0 - t_val) * v                            # (N,1,T_x)

            # (1) INNER LOOP — shared (w, α, φ); skip if 'fixed'
            if init_mode == 'fixed':
                h_est = h_frozen                                       # (1,1,h_length)
            else:
                x_det = x_true.detach() if use_oracle_x else x_hat0.detach()
                sigma_h = sigma_h_max * (1.0 - t_val) + sigma_h_min * t_val
                for inner_it in range(n_its):
                    h_param  = _build_h()                              # (1,1,h_length)
                    y_pred_h = grouped_conv_full_sharedh(x_det, h_param)  # (N,1,T_y)
                    # sums over N AND (freq,frame) — the N-fold constraint on
                    # the shared STFT IR. One backward aggregates onto (w,α,φ).
                    L_data   = stft_mag_compress_loss(y, y_pred_h, c_compress,
                                                      stft_win, stft_hop, window)
                    L_reg    = sigma_h * (h_param * torch.randn_like(h_param)).sum()
                    L_total  = L_data + L_reg
                    opt_h.zero_grad()
                    L_total.backward()

                    # ---- DIAGNOSTIC (instrumentation only) ----------------
                    # Fires once (k=0, inner_it=0) when debug_grad=True.
                    # Verifies the shared (w,α,φ) gradient aggregates ALL N
                    # claps. Save/restore so opt_h.step() uses the real grad.
                    if debug_grad and k == 0 and inner_it == 0:
                        print(f"  [diagnostic] N={N}, k=0, inner_it=0")
                        print(f"    aggregated grad norms (real grad opt_h.step() will use):")
                        print(f"      w_raw.grad.norm()     = {w_raw.grad.norm().item():.6e}")
                        print(f"      alpha_raw.grad.norm() = {alpha_raw.grad.norm().item():.6e}")
                        print(f"      phi.grad.norm()       = {phi.grad.norm().item():.6e}")
                        saved_w   = w_raw.grad.detach().clone()
                        saved_a   = alpha_raw.grad.detach().clone()
                        saved_phi = phi.grad.detach().clone()
                        print(f"    per-clap contribution decomposition:")
                        for i in range(N):
                            w_raw.grad = None; alpha_raw.grad = None; phi.grad = None
                            h_param_i = _build_h()
                            y_pred_i = grouped_conv_full_sharedh(x_det[i:i+1], h_param_i)
                            L_i = stft_mag_compress_loss(y[i:i+1], y_pred_i, c_compress,
                                                         stft_win, stft_hop, window)
                            L_i.backward()
                            print(f"      clap {i+1}/{N}: "
                                  f"w={w_raw.grad.norm().item():.6e}, "
                                  f"α={alpha_raw.grad.norm().item():.6e}, "
                                  f"φ={phi.grad.norm().item():.6e}")
                        w_raw.grad = saved_w; alpha_raw.grad = saved_a; phi.grad = saved_phi
                        print(f"    aggregated grads restored — real Adam step proceeds")
                    # ---- END DIAGNOSTIC ------------------------------------

                    # phase-freedom annealing: freeze phi until t_val >= threshold
                    if t_val < phase_unfreeze_after and phi.grad is not None:
                        phi.grad.zero_()
                    opt_h.step()

                with torch.no_grad():
                    h_est = _build_h().detach()                        # (1,1,h_length)

            # (2) OUTER STEP — x guidance, shared h held constant
            y_pred = grouped_conv_full_sharedh(x_hat0, h_est)          # (N,1,T_y)
            L_x = stft_mag_compress_loss(y, y_pred, c_compress,
                                         stft_win, stft_hop, window)
            g = torch.autograd.grad(L_x, x)[0]                         # (N,1,T_x)

            with torch.no_grad():
                g_norm = g.flatten(1).norm(dim=1).view(-1, 1, 1)
                g_dir  = g / (g_norm + 1e-8)
                x = x + h_step * v - xi_x * g_dir
                if t_val > peak_norm_after:
                    peak = x.abs().amax(dim=(-1, -2), keepdim=True)
                    x = x / (peak + 1e-8)

        with torch.no_grad():
            w_final     = F.softplus(w_raw).detach()                  # (1, F)
            alpha_final = F.softplus(alpha_raw).detach()              # (1, F)
            phi_final   = phi.detach()                                # (1, F, T_frames)
        return x.detach(), h_est.detach(), w_final, alpha_final, phi_final


# ---- helpers for sample_blind_wiener (module-level for testability) ----

def apply_direct_path_constraint(h_est: torch.Tensor, tiny: float = 1e-6) -> torch.Tensor:
    """
    BUDDy δ⊕ in spirit: direct path is the unit anchor, everything else is
    relative. Rescale h_est by h_est[..., 0:1] when |h[0]| is non-tiny;
    otherwise leave the tail unchanged and clamp h[0] = 1 as fallback.
    After this op, h_est[..., 0] == 1 for every batch element.
    """
    h0      = h_est[..., 0:1]                               # (..., 1)
    safe    = h0.abs() > tiny
    divisor = torch.where(safe, h0, torch.ones_like(h0))
    h_est   = h_est / divisor                               # rescaled where safe, unchanged where not
    # unconditional h[0] = 1 (no-op when rescaled, fallback when not)
    h_est   = torch.cat([torch.ones_like(h0), h_est[..., 1:]], dim=-1)
    return h_est


def clamp_tail_energy(h_est: torch.Tensor, tail_max_norm: float) -> torch.Tensor:
    """
    Cap ||h[..., 1:]||_2 per sample at tail_max_norm. Direction preserved
    (scaled down uniformly when over cap, untouched when under). Direct-path
    sample h[..., 0] is NOT touched — caller is responsible for invoking
    apply_direct_path_constraint first so h[0]=1 stays an anchor.

    Order matters: apply this AFTER apply_direct_path_constraint, not
    before. Rescaling by h[0] would otherwise undo the clamp.
    """
    tail = h_est[..., 1:]
    tail_norm = tail.flatten(1).norm(dim=1).view(-1, 1, 1)        # (B,1,1)
    scale = torch.clamp(tail_max_norm / (tail_norm + 1e-8), max=1.0)
    tail = tail * scale
    return torch.cat([h_est[..., 0:1], tail], dim=-1)


# ---- helpers for sample_blind_structured (B2-time: BUDDy-style structured IR) ----

def build_structured_ir(w: torch.Tensor,
                        alpha: torch.Tensor,
                        u: torch.Tensor,
                        h_length: int,
                        device) -> torch.Tensor:
    """
    Differentiable structured IR:  h(n) = δ(n) + w · exp(-α·n) · u(n).

    Inputs (all with grad if you want grad to flow):
        w     : (B, 1, 1)         tail gain (positive — caller is responsible
                                  for keeping it positive, e.g. softplus)
        alpha : (B, 1, 1)         decay rate per sample (positive)
        u     : (B, 1, h_length)  free tail-detail vector

    The cat-based construction discards tail[..., 0] = w·u[0] in favour of
    pinning h[0] = 1 (BUDDy δ⊕). This is intentional — direct-path is the
    unit anchor; the discard is the cost of that anchor.

    No in-place ops (would break autograd to w, α, u via tail[1:]).
    """
    n   = torch.arange(h_length, device=device, dtype=w.dtype).view(1, 1, -1)
    env = w * torch.exp(-alpha * n)                       # (B, 1, h_length)
    tail = env * u                                        # (B, 1, h_length)
    ones = torch.ones_like(tail[..., 0:1])                # (B, 1, 1)
    return torch.cat([ones, tail[..., 1:]], dim=-1)       # tail[0] discarded — intentional


# ============================================================================
# B2-stft: STFT-domain structured IR parameterization
# ============================================================================
# Time-domain B2-time caps env_corr(h) at ~0.43 even with oracle-x at N=8: the
# 1600-dim noise vector u is unidentifiable (its phase/realization stays
# entangled with x's phase — see clapgen_progress_log.md §5c). The STFT
# parameterization replaces u with a STRUCTURED per-bin magnitude envelope plus
# a FREE per-bin phase φ[k,n]. Free phase is the lever the N-clap complementary
# constraints can actually pin down.

def stft_compress(Z: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    """
    BUDDy magnitude compression in the STFT domain.

        comp(Z) = |Z|^c · exp(j·∠Z)

    Implemented as Z · |Z|^(c-1) (preserves phase exactly, scales magnitude to
    |Z|^c). |Z| is clamped at `eps` so the c<1 power's gradient stays finite
    near silent bins. Eloi STRONGLY recommends this (c=0.5 or lower) — it
    de-emphasizes the loud direct path and balances the gradient across the
    quieter reverb tail, which is where the IR detail lives.

    Z must be complex; returns a complex tensor of the same shape.
    """
    mag = Z.abs().clamp(min=eps)
    return Z * mag.pow(c - 1.0)


def stft_mag_compress_loss(y: torch.Tensor,
                           y_pred: torch.Tensor,
                           c: float,
                           n_fft: int,
                           hop_length: int,
                           window: torch.Tensor) -> torch.Tensor:
    """
    Magnitude-compressed STFT-domain likelihood:

        L = || comp(STFT(y)) − comp(STFT(y_pred)) ||²

    y, y_pred : (B, 1, T_y)  time-domain observation / reconstruction
    Returns a scalar (summed over batch, freq, frame, real+imag). For the
    shared/multi-clap form the sum over the batch dim is exactly the N-fold
    constraint on the shared IR.
    """
    Y  = torch.stft(y.squeeze(1),      n_fft=n_fft, hop_length=hop_length,
                    win_length=n_fft, window=window, center=True,
                    return_complex=True, onesided=True)
    Yp = torch.stft(y_pred.squeeze(1), n_fft=n_fft, hop_length=hop_length,
                    win_length=n_fft, window=window, center=True,
                    return_complex=True, onesided=True)
    d = stft_compress(Y, c) - stft_compress(Yp, c)
    return (d.real ** 2 + d.imag ** 2).sum()


def build_stft_ir(w: torch.Tensor,
                  alpha: torch.Tensor,
                  phi: torch.Tensor,
                  h_length: int,
                  n_fft: int,
                  hop_length: int,
                  window: torch.Tensor,
                  frame_idx: torch.Tensor) -> torch.Tensor:
    """
    Differentiable STFT-domain structured IR. The STFT analog of
    build_structured_ir — same delta-plus direct-path anchor, but the tail is
    parameterized in the STFT domain so the phase is free per bin.

        |H[k, n]| = w[k] · exp(-α[k] · n)        per-bin exponential envelope
         ∠H[k, n] = φ[k, n]                       FREE per-bin phase
        h_time    = iSTFT(H)
        h         = δ⊕  →  h[0] = 1, tail = h_time[1:]   (delta-plus anchor)

    Inputs (all may carry grad):
        w         : (B, F)            per-bin tail gain   (positive — softplus upstream)
        alpha     : (B, F)            per-bin decay rate  (positive — softplus upstream)
        phi       : (B, F, T_frames)  free per-bin phase
        frame_idx : (T_frames,)       0,1,...,T_frames-1  (envelope is over frames)

    Returns h : (B, 1, h_length).

    The phase φ is what time-domain `u` could not capture: per-bin, per-frame
    free phase that the convolution observation can constrain (especially with
    complementary multi-clap spectra). No in-place ops (autograd to w, α, φ).
    """
    mag = w.unsqueeze(-1) * torch.exp(-alpha.unsqueeze(-1) * frame_idx.view(1, 1, -1))
    H   = torch.polar(mag, phi)                       # (B, F, T_frames) complex
    h_time = torch.istft(H, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                         window=window, center=True, length=h_length,
                         onesided=True, return_complex=False)   # (B, h_length)
    ones = torch.ones_like(h_time[..., :1])           # (B, 1)
    h    = torch.cat([ones, h_time[..., 1:]], dim=-1)  # delta-plus: h[0]=1, tail discarded at 0
    return h.unsqueeze(1)                              # (B, 1, h_length)


def grouped_conv_full_diffh(x_det: torch.Tensor,
                            h_param: torch.Tensor) -> torch.Tensor:
    """
    Same op as grouped_conv_full but conventionally used in the INNER loop:
    x_det is detached (constant); h_param carries grad to (w, α, u).
    F.conv1d differentiates w.r.t. whichever inputs have requires_grad=True;
    the function body is identical to grouped_conv_full.
    """
    return grouped_conv_full(x_det, h_param)


def grouped_conv_full_sharedh(x: torch.Tensor,
                              h_shared: torch.Tensor) -> torch.Tensor:
    """
    Per-sample 1D convolution with a SINGLE SHARED IR. Variant-B form.

    x        : (N, 1, T_x)         N independent claps (may carry grad)
    h_shared : (1, 1, h_length)    one IR — used for every batch element
    returns  : (N, 1, T_x + h_length - 1)

    Semantically equivalent to:
        h_bcast = h_shared.expand(N, 1, h_length)
        grouped_conv_full(x, h_bcast)
    but reuses the same h_shared tensor in each of the N conv calls instead
    of materializing an expanded copy. PyTorch's autograd correctly
    aggregates: gradient on h_shared = sum over i of grad through conv1d_i,
    which IS the N-fold-constraint mechanism for the shared-IR likelihood.

    Differentiable w.r.t. both x and h_shared.
    """
    L_h = h_shared.shape[-1]
    h_flipped = h_shared.flip(-1)                # (1, 1, h_length)
    outs = [
        F.conv1d(x[i:i+1], h_flipped, padding=L_h - 1)
        for i in range(x.shape[0])
    ]
    return torch.cat(outs, dim=0)


def fit_structured_ir_to_target(target_h: torch.Tensor,
                                h_length: int,
                                batch_size: int,
                                device,
                                init_w_value: float = 0.4,
                                init_alpha_value: float = 7.8125e-4,
                                n_iters: int = 200,
                                lr: float = 0.1):
    """
    Offline Adam fit of (w_raw, alpha_raw, u) so that
        build_structured_ir(softplus(w_raw), softplus(alpha_raw), u) ≈ target_h
    in MSE. Returns the three leaves detached, ready to be used as init for
    sample_blind_structured's 'fixed' or 'anchored' modes.

    target_h is broadcast to batch_size; B independent fits but all to the
    same target. (w, α) should converge to similar values across batch; u
    diverges per element (different random init).
    """
    import math
    inv_sp = lambda y: math.log(math.expm1(y))            # inverse softplus

    w_raw = torch.full((batch_size, 1, 1), inv_sp(init_w_value),
                       device=device, dtype=torch.float32, requires_grad=True)
    alpha_raw = torch.full((batch_size, 1, 1), inv_sp(init_alpha_value),
                           device=device, dtype=torch.float32, requires_grad=True)
    u = torch.randn(batch_size, 1, h_length, device=device, requires_grad=True)

    target = target_h.view(1, 1, -1).expand(batch_size, 1, -1).contiguous()

    opt = torch.optim.Adam([w_raw, alpha_raw, u], lr=lr)
    for _ in range(n_iters):
        w     = F.softplus(w_raw)
        alpha = F.softplus(alpha_raw)
        h     = build_structured_ir(w, alpha, u, h_length, device)
        loss  = ((h - target) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return w_raw.detach(), alpha_raw.detach(), u.detach()


def wiener_update(x_hat0: torch.Tensor,
                  y: torch.Tensor,
                  h_length: int,
                  eps: float) -> torch.Tensor:
    """
    Per-sample Tikhonov-regularized frequency-domain deconvolution.

    Solve  H = conj(X) * Y / (|X|² + eps)  per FFT bin, then irfft and
    truncate to h_length. Apply the direct-path constraint at the end.

    Args:
        x_hat0 : (B, 1, T_x)  data estimate (already detached upstream)
        y      : (B, 1, T_y)  observation
        h_length: int         desired IR length
        eps    : float        Tikhonov regularizer (typically time-dependent)

    Returns:
        h_est  : (B, 1, h_length)  per-sample IR estimate, h_est[..., 0] == 1
    """
    T_x = x_hat0.shape[-1]
    # next pow2 >= T_x + h_length - 1 (covers linear-conv support)
    n_lin = T_x + h_length - 1
    n_fft = 1 << ((n_lin - 1).bit_length())

    X = torch.fft.rfft(x_hat0, n=n_fft, dim=-1)
    Y = torch.fft.rfft(y,      n=n_fft, dim=-1)

    H = (X.conj() * Y) / (X.abs() ** 2 + eps)
    h_full = torch.fft.irfft(H, n=n_fft, dim=-1)            # (B, 1, n_fft)
    h_est  = h_full[..., :h_length]                         # (B, 1, h_length)

    return apply_direct_path_constraint(h_est)


def grouped_conv_full(x: torch.Tensor, h_est: torch.Tensor) -> torch.Tensor:
    """
    Per-sample true 1D convolution (loop, NOT groups=B — cleaner autograd).

    x      : (B, 1, T_x)        differentiable input
    h_est  : (B, 1, h_length)   treated as constant (do not require grad)
    returns: (B, 1, T_x + h_length - 1)

    Each iteration's slice [i:i+1] is a view of x that retains grad_fn,
    so the gradient flows back to x through the cat.
    """
    L_h = h_est.shape[-1]
    outs = [
        F.conv1d(x[i:i+1], h_est[i:i+1].flip(-1), padding=L_h - 1)
        for i in range(x.shape[0])
    ]
    return torch.cat(outs, dim=0)


# ---- time embedding ----
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
        args = t[:, None] * freqs[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)  # (B, out_dim)


# ---- FiLM residual block (dilated conv) ----
class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, t_dim: int, dilation: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

        self.film = nn.Linear(t_dim, 2 * channels)

    def forward(self, x: torch.Tensor, t_ctx: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        gb = self.film(t_ctx)           # (B, 2C)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        gamma = gamma[:, :, None]       # (B, C, 1)
        beta = beta[:, :, None]         # (B, C, 1)

        h = self.norm2(h)
        h = h * (1.0 + gamma) + beta
        h = self.conv2(F.silu(h))

        return x + h


def receptive_field(dilations):
    """
    Total receptive field of a stack of DilatedResBlocks. Each block has
    conv1 (k=3, dilation=d) followed by conv2 (k=3, dilation=1), so the
    block contributes 2*d + 2 to the RF; the initial token counts as 1.
    """
    rf = 1
    for d in dilations:
        rf += 2 * d + 2
    return rf


# ---- small velocity network for 960-sample clips ----
class WaveNetVelocity(nn.Module):
    """
    x: (B, 1, T) raw waveform
    t: (B,) continuous time (e.g., in [0,1])
    returns v: (B, 1, T)
    """
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        num_blocks: int = 10,
        t_embed_dim: int = 128,
        t_hidden_dim: int = 256,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(t_embed_dim)
        self.time_mlp = TimeMLP(t_embed_dim, t_hidden_dim)

        self.in_proj = nn.Conv1d(in_channels, channels, kernel_size=1)

        # dilations: 1, 2, 4, ..., 2**(num_blocks-1). Monotonic doubling — no
        # cycling — so RF grows exponentially with num_blocks and covers the
        # whole signal at high fs / long T.
        self.dilations = [2 ** i for i in range(num_blocks)]
        self.receptive_field = receptive_field(self.dilations)

        blocks = []
        for d in self.dilations:
            blocks.append(DilatedResBlock(channels, t_hidden_dim, dilation=d))
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.out_proj = nn.Conv1d(channels, in_channels, kernel_size=1)

        print(f"WaveNetVelocity: channels={channels}, num_blocks={num_blocks}, "
              f"dilations={self.dilations}, receptive_field={self.receptive_field}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ctx = self.time_mlp(self.time_emb(t))  # (B, t_hidden_dim)

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_ctx)

        v = self.out_proj(F.silu(self.out_norm(h)))
        return v


if __name__ == "__main__":
    B, T = 8, 960  # 20 ms at 48 kHz
    x = torch.randn(B, 1, T)
    t = torch.rand(B)

    model = WaveNetVelocity(channels=64, num_blocks=10)
    v = model(x, t)
    print(v.shape)  # (8, 1, 960)