from __future__ import annotations

from importlib import import_module

__all__ = [
    "SortedMidiNote",
    "extract_sorted_notes",
    "replace_note_velocities",
    "write_flat_velocity_copy",
    "VelocityDistributionMapping",
    "combine_note_loudness_to_percentiles",
    "map_note_loudness_to_midi_velocity",
    "NoteFeatureResult",
    "NoteLoudnessConfig",
    "extract_note_loudness_features",
    "extract_note_loudness_from_files",
    "Route1Config",
    "predict_direct_inversion_for_pair",
    "predict_route1_dataset",
    "predict_flat_dataset",
    "Route1EvalJobRequest",
    "Route1EvalJobResult",
    "run_route1_eval_job",
    "FlatEvalJobRequest",
    "FlatEvalJobResult",
    "run_flat_eval_job",
    "Route2EvalJobRequest",
    "Route2EvalJobResult",
    "run_route2_eval_job",
    "predict_route2_dataset",
    "predict_route3_dataset",
    "predict_route4_dataset",
    "PredictionItem",
    "attach_reference_audio_from_folder",
    "build_dataset_prediction_manifest",
    "build_folder_prediction_manifest",
    "evaluate_prediction_item",
    "evaluate_prediction_manifest",
    "evaluation_results_to_dataframe",
]

_EXPORTS = {
    "SortedMidiNote": (".common", "SortedMidiNote"),
    "extract_sorted_notes": (".common", "extract_sorted_notes"),
    "replace_note_velocities": (".common", "replace_note_velocities"),
    "write_flat_velocity_copy": (".common", "write_flat_velocity_copy"),
    "VelocityDistributionMapping": (".route1_infer", "VelocityDistributionMapping"),
    "combine_note_loudness_to_percentiles": (".route1_infer", "combine_note_loudness_to_percentiles"),
    "map_note_loudness_to_midi_velocity": (".route1_infer", "map_note_loudness_to_midi_velocity"),
    "NoteFeatureResult": (".route1_infer", "NoteFeatureResult"),
    "NoteLoudnessConfig": (".route1_infer", "NoteLoudnessConfig"),
    "extract_note_loudness_features": (".route1_infer", "extract_note_loudness_features"),
    "extract_note_loudness_from_files": (".route1_infer", "extract_note_loudness_from_files"),
    "Route1Config": (".route1_infer", "Route1Config"),
    "predict_direct_inversion_for_pair": (".route1_infer", "predict_direct_inversion_for_pair"),
    "predict_route1_dataset": (".route1_infer", "predict_route1_dataset"),
    "predict_flat_dataset": (".flat_infer", "predict_flat_dataset"),
    "Route1EvalJobRequest": (".route1_eval_job", "Route1EvalJobRequest"),
    "Route1EvalJobResult": (".route1_eval_job", "Route1EvalJobResult"),
    "run_route1_eval_job": (".route1_eval_job", "run_route1_eval_job"),
    "FlatEvalJobRequest": (".flat_eval_job", "FlatEvalJobRequest"),
    "FlatEvalJobResult": (".flat_eval_job", "FlatEvalJobResult"),
    "run_flat_eval_job": (".flat_eval_job", "run_flat_eval_job"),
    "Route2EvalJobRequest": (".route2_eval_job", "Route2EvalJobRequest"),
    "Route2EvalJobResult": (".route2_eval_job", "Route2EvalJobResult"),
    "run_route2_eval_job": (".route2_eval_job", "run_route2_eval_job"),
    "predict_route2_dataset": (".route2_infer", "predict_route2_dataset"),
    "predict_route3_dataset": (".route3_infer", "predict_route3_dataset"),
    "predict_route4_dataset": (".route4_infer", "predict_route4_dataset"),
    "PredictionItem": (".eval_framework", "PredictionItem"),
    "attach_reference_audio_from_folder": (".eval_framework", "attach_reference_audio_from_folder"),
    "build_dataset_prediction_manifest": (".eval_framework", "build_dataset_prediction_manifest"),
    "build_folder_prediction_manifest": (".eval_framework", "build_folder_prediction_manifest"),
    "evaluate_prediction_item": (".eval_framework", "evaluate_prediction_item"),
    "evaluate_prediction_manifest": (".eval_framework", "evaluate_prediction_manifest"),
    "evaluation_results_to_dataframe": (".eval_framework", "evaluation_results_to_dataframe"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
