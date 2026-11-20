import re
from pathlib import Path

from mantid.kernel import logger

from mvesuvio.util.files_manager import FilesManager


def get_bootstrap_input_directories(inputs_dir_path: Path):
    if not inputs_dir_path.is_dir():
        logger.error("The inputs directory path provided for bootstrap is not a directory.")
        return ()

    inputs_backward_path = inputs_dir_path / "backward"
    if not inputs_backward_path.exists():
        inputs_backward_path.mkdir(exist_ok=True)
        logger.error(f"Created backward directory. Please place your backward samples here: {inputs_backward_path!s}")
        return ()

    inputs_forward_path = inputs_dir_path / "forward"
    if not inputs_forward_path.exists():
        inputs_forward_path.mkdir(exist_ok=True)
        logger.error(f"Created forward directory. Please place your forward samples here: {inputs_forward_path!s}")
        return ()

    return (inputs_dir_path, inputs_backward_path, inputs_forward_path)


def get_common_prefix_of_bootstrap_sample_names(back_ws_name: str, front_ws_name: str) -> str:
    if back_ws_name == front_ws_name:
        logger.error(f"Bootstrap Error: backward and forward inputs should not have the same name: {front_ws_name}.")
        return ""

    if back_ws_name[-1] != front_ws_name[-1]:
        logger.error(f"Bootstrap Error: inputs {back_ws_name} and {front_ws_name} do not have the same last character.")
        return ""

    def longest_common_prefix(s1: str, s2: str) -> str:
        return s1[: next((i for i, (a, b) in enumerate(zip(s1, s2)) if a != b or a == "_" or b == "_"), min(len(s1), len(s2)))]

    common_prefix = longest_common_prefix(back_ws_name, front_ws_name)
    if not common_prefix:
        logger.error(f"Bootstrap Error: inputs {back_ws_name} and {front_ws_name} do not have a common prefix.")
        return ""
    return common_prefix


def get_bootstrap_sample_sort_key(path_obj: Path) -> tuple[int, str]:
    numeric_matches = re.findall(r"(\d+)", path_obj.stem)
    if numeric_matches:
        return (int(numeric_matches[-1]), path_obj.name)
    return (0, path_obj.name)


def pair_bootstrap_sample_paths(back_paths: list[Path], front_paths: list[Path]) -> list[tuple[Path, Path]]:
    back_sorted = sorted(back_paths, key=get_bootstrap_sample_sort_key)
    front_sorted = sorted(front_paths, key=get_bootstrap_sample_sort_key)

    if len(back_sorted) != len(front_sorted):
        logger.warning(
            "Bootstrap input counts do not match: backward=%d, forward=%d. Processing only the matching sample pairs.",
            len(back_sorted),
            len(front_sorted),
        )

    common_count = min(len(back_sorted), len(front_sorted))
    return list(zip(back_sorted[:common_count], front_sorted[:common_count]))


def apply_bootstrap_overrides(back_inputs, front_inputs, back_ws_path: Path | None, front_ws_path: Path | None) -> None:
    back_override = str(back_ws_path.absolute()) if back_ws_path is not None else ""
    front_override = str(front_ws_path.absolute()) if front_ws_path is not None else ""

    # Keep legacy attribute for compatibility while using the new canonical override field.
    back_inputs.override_input_workspace = back_override
    front_inputs.override_input_workspace = front_override
    back_inputs.overwrite_analysis_input_workspace = back_override
    front_inputs.overwrite_analysis_input_workspace = front_override

    if back_override:
        back_inputs.name = Path(back_override).stem
    if front_override:
        front_inputs.name = Path(front_override).stem

    # Keep bootstrap execution non-interactive on repeated samples.
    if hasattr(back_inputs, "show_plots"):
        back_inputs.show_plots = False
    if hasattr(front_inputs, "show_plots"):
        front_inputs.show_plots = False


def update_sample_inputs_outputs(
    back_inputs, front_inputs, back_ws_path: Path | None, front_ws_path: Path | None, output_path: Path
) -> None:
    output_path.mkdir(exist_ok=True)
    apply_bootstrap_overrides(back_inputs, front_inputs, back_ws_path, front_ws_path)
    FilesManager.set_experiment_dir(output_path)
