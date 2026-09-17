# Provenance

Every file in this repository and where it came from in the research tree.
Contents are verbatim; only import paths and repository-root depths were
rewritten, because the files moved.

| original (clapgen) | this repository |
|---|---|
| `scripts/queue/arpege_external_eval.py` | `experiments/arpege_external_evaluation.py` |
| `scripts/data/arpege_pairing.py` | `experiments/arpege_pairing.py` |
| `scripts/queue/cropped_excitation_6ms.py` | `experiments/cropped_excitation_baseline.py` |
| `scripts/queue/cropped_excitation_6ms_report.py` | `experiments/cropped_excitation_baseline_report.py` |
| `scripts/queue/cropped_excitation_6ms_validate.py` | `experiments/cropped_excitation_baseline_validate.py` |
| `scripts/queue/edc_broadband_norm_column.py` | `experiments/energy_decay_broadband_column.py` |
| `scripts/queue/edc_dalsanto_isolation.py` | `experiments/energy_decay_metric_isolation.py` |
| `scripts/queue/edc_dalsanto_noisy_addendum.py` | `experiments/energy_decay_metric_noisy_rows.py` |
| `scripts/queue/edc_dalsanto_recompute.py` | `experiments/energy_decay_metric_recompute.py` |
| `scripts/queue/e1_e2_recoverability.py` | `experiments/excitation_recoverability.py` |
| `scripts/queue/flow_audit_report.py` | `experiments/flow_mechanism_audit.py` |
| `scripts/queue/e3_matched_comparison.py` | `experiments/matched_estimator_comparison.py` |
| `scripts/queue/e3_report.py` | `experiments/matched_estimator_report.py` |
| `scripts/queue/noisy_benchmark_common.py` | `experiments/noisy_benchmark_common.py` |
| `scripts/queue/noisy_regression_train.py` | `experiments/noisy_benchmark_train.py` |
| `scripts/queue/e6_phone_deployment.py` | `experiments/phone_deployment_evaluation.py` |
| `scripts/queue/e5_spheres_real_clap.py` | `experiments/real_clap_room_evaluation.py` |
| `scripts/queue/shoebox_true_1s_checkpoint_audit.py` | `experiments/shoebox_checkpoint_audit.py` |
| `scripts/queue/regenerate_shoebox_1s_shard.py` | `experiments/shoebox_one_second_shard.py` |
| `scripts/queue/shoebox_one_second_support.py` | `experiments/shoebox_one_second_support.py` |
| `scripts/queue/shoebox_1s_retrained_evaluate.py` | `experiments/shoebox_retrained_evaluation.py` |
| `scripts/queue/post_meeting_table1.py` | `experiments/single_clap_benchmark.py` |
| `scripts/queue/table1_mean_std.py` | `experiments/single_clap_benchmark_statistics.py` |
| `scripts/figures/arpege_external_figures.py` | `figures/arpege_external_figures.py` |
| `scripts/figures/paper_figures.py` | `figures/benchmark_comparison_figures.py` |
| `scripts/figures/export_current_benchmark_examples.py` | `figures/export_benchmark_examples.py` |
| `scripts/figures/export_model_comparisons.py` | `figures/export_model_comparisons.py` |
| `scripts/figures/export_e5_examples.py` | `figures/export_real_clap_examples.py` |
| `scripts/figures/clap_dataset_figure.py` | `figures/handclap_dataset_figure.py` |
| `scripts/figures/dataset_statistics.py` | `figures/handclap_dataset_statistics.py` |
| `scripts/figures/model_comparison_figures.py` | `figures/model_comparison_figures.py` |
| `scripts/figures/phone_qualitative_compact.py` | `figures/phone_qualitative_figure.py` |
| `scripts/figures/phone_recording_context.py` | `figures/phone_recording_context.py` |
| `scripts/figures/phone_spectral_review.py` | `figures/phone_spectral_analysis.py` |
| `scripts/figures/phone_spectrogram_display_review.py` | `figures/phone_spectrogram_figure.py` |
| `scripts/figures/e5_e6_figures.py` | `figures/real_clap_and_phone_figures.py` |
| `scripts/figures/recoverability_figure.py` | `figures/recoverability_figure.py` |
| `src/clapgen/experiments/clap_spectrum_audit/run.py` | `src/claprir/analysis/clap_spectrum.py` |
| `src/clapgen/experiments/deconvolution_audit/run.py` | `src/claprir/analysis/regularization_selection.py` |
| `src/clapgen/experiments/supervised_comparison/materialize.py` | `src/claprir/datasets/controlled_shards.py` |
| `src/clapgen/experiments/real_clap_multiclap/extract.py` | `src/claprir/datasets/handclap_corpus.py` |
| `src/clapgen/experiments/multiroom_generalization/manifest.py` | `src/claprir/datasets/rir_shards.py` |
| `src/clapgen/models/supervised.py` | `src/claprir/datasets/shoebox_rirs.py` |
| `src/clapgen/evaluation/metrics.py` | `src/claprir/metrics/deconvolution.py` |
| `src/clapgen/evaluation/edc_dalsanto.py` | `src/claprir/metrics/energy_decay.py` |
| `eloi_flow_debug/lundeby.py` | `src/claprir/metrics/lundeby_truncation.py` |
| `src/clapgen/evaluation/diagnosis.py` | `src/claprir/metrics/room_acoustics.py` |
| `src/clapgen/models/direct_rir.py` | `src/claprir/models/direct_rir_estimator.py` |
| `src/clapgen/models/supervised_estimators.py` | `src/claprir/models/excitation_estimator.py` |
| `eloi_flow_debug/flow_sampler.py` | `src/claprir/models/flow_sampler.py` |
| `src/clapgen/models/hybrid_rir.py` | `src/claprir/models/hybrid_rir_estimator.py` |
| `src/clapgen/models/ncsnpp/` | `src/claprir/models/ncsnpp/` |
| `src/clapgen/models/representations.py` | `src/claprir/models/representations.py` |
| `src/clapgen/models/residual_early.py` | `src/claprir/models/residual_early_estimator.py` |
| `src/clapgen/core/network.py` | `src/claprir/models/spectrogram_network.py` |
| `src/clapgen/models/unet_1d.py` | `src/claprir/models/time_unet.py` |
| `src/clapgen/evaluation/figure_style.py` | `src/claprir/plotting/style.py` |
| `src/clapgen/rir/metadata.py` | `src/claprir/rir/metadata.py` |
| `src/clapgen/rir/providers.py` | `src/claprir/rir/providers.py` |
| `src/clapgen/rir/registry.py` | `src/claprir/rir/registry.py` |
| `src/clapgen/rir/signals.py` | `src/claprir/rir/signals.py` |
| `src/clapgen/rir/sofa.py` | `src/claprir/rir/sofa.py` |
| `src/clapgen/experiments/hybrid_direct_rir/run.py` | `src/claprir/training/train_rir_estimator.py` |
| `publication/paper/build_tables.py` | `tables/build_tables.py` |
