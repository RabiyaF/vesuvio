from pathlib import Path
from shutil import copyfile, copytree, ignore_patterns

PACKAGE_CONFIG_PATH = Path(__file__).absolute().parent.with_name("default_config")
USER_CONFIG_PATH = Path.home() / "mvesuvio"
VESUVIO_PROPERTIES_PATH = PACKAGE_CONFIG_PATH / "vesuvio.user.properties"
PLOTS_CONFIG_PATH = PACKAGE_CONFIG_PATH / "vesuvio.plots.mplstyle"

COPY_IF_NOT_PRESENT = {"ip_files": "dir", "experiment_template/script_to_create_figures.py": "file", "vesuvio.plots.mplstyle": "file"}

ALWAYS_COPY = {
    "experiment_template/run_reduction.py": "file",
    "experiment_template/run_fitting.py": "file",
}


def set_default_config_vars():
    set_config_vars(
        {
            "caching.inputs": str(USER_CONFIG_PATH / "experiment_template"),
            "caching.ipfolder": str(USER_CONFIG_PATH / "ip_files"),
        }
    )


def _read_config(config_file_path, throw_on_not_found=True):
    lines = ""
    try:
        with open(config_file_path, "r") as file:
            lines = file.readlines()
    except OSError:
        if throw_on_not_found:
            raise RuntimeError(f"Could not read from vesuvio config file: {config_file_path}")
    return lines


def set_config_vars(var_dict):
    file_path = VESUVIO_PROPERTIES_PATH
    lines = _read_config(file_path)

    updated_lines = []
    for line in lines:
        match = False
        for var in var_dict:
            if line.startswith(var):
                new_line = f"{var}={var_dict[var]}"
                updated_lines.append(f"{new_line}\n")
                match = True
                print(f"Setting: {new_line}")
                break

        if not match:
            updated_lines.append(line)

    with open(file_path, "w") as file:
        file.writelines(updated_lines)


def read_cached_var(var, throw_on_not_found=True):
    lines = _read_config(VESUVIO_PROPERTIES_PATH, throw_on_not_found)

    result = ""
    for line in lines:
        if line.startswith(var):
            result = line.split("=", 2)[1].strip("\n")
            break
    if not result and throw_on_not_found:
        raise ValueError(f"{var} was not found in the vesuvio config")
    return result


def get_experiment_name():
    return Path(read_cached_var("caching.inputs")).name


def get_plots_config_file() -> str:
    return str(PLOTS_CONFIG_PATH)


def refresh_config_dir_and_contents():
    USER_CONFIG_PATH.mkdir(exist_ok=True)
    _copy_config_entries(COPY_IF_NOT_PRESENT, overwrite=False)
    _copy_config_entries(ALWAYS_COPY, overwrite=True)


def _copy_config_entries(entries, overwrite=False):
    for relative_path, entry_type in entries.items():
        destination = Path(USER_CONFIG_PATH, relative_path)

        if not overwrite and destination.exists():
            continue

        _copy_config_entry(relative_path, entry_type, overwrite)


def _copy_config_entry(relative_path, entry_type, overwrite):
    source = Path(PACKAGE_CONFIG_PATH, relative_path)
    destination = Path(USER_CONFIG_PATH, relative_path)

    if entry_type == "file":
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
        return

    if entry_type == "dir":
        destination.parent.mkdir(parents=True, exist_ok=True)
        copytree(source, destination, dirs_exist_ok=overwrite, ignore=ignore_patterns("__*"))
        return

    raise ValueError(f"Unknown config entry type: {entry_type}")


def is_cache_set():
    return bool(read_cached_var("caching.inputs", False))
