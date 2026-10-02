from typing import Any, Dict, List, Set

from runs.config_general import ArgsGeneralDefaults


def override_config(
    config: ArgsGeneralDefaults,
    config_override: Dict[str, Any],
    missing_keys_skip: List[str] | Set[str] = [],
    keys_dont_override: List[str] | Set[str] = [],
):
    """
    Overrides the config with the values from the config_override dict. The missing_keys_skip
    list specifies keys in the config that will not be overridden by the pretrain config. An
    error is raised if any keys in the config (except the ones in missing_keys_skip) are
    missing from the config_override dict.
    """
    # Get a list of the default keys of the pretrain config.
    config_default_keys: List[str] = list(
        config._get_argument_names()  # pylint: disable=W0212
    )

    #### Check for missing keys ####
    # We search for missing keys from the config to override that we expect to be in the given config.
    # First we fetch the required keys from the pretrain config, as they need to be present.
    required_keys = [
        key
        for key in config_default_keys
        if key not in config._get_class_dict().keys()  # pylint: disable=W0212
    ]
    # The .get_class_dict() method above returns a dictionary of the class attributes with default values assigned.
    # Thus if a key is not present in that dict, it is required.

    # We exclude the general keys and optionally keys specified in missing_keys_skip.
    # Note that if a required key is specified in missing_keys_skip, the code will check
    # for it anyhow, otherwise it will raise an error when calling .from_dict() below.
    # We also dont check for the keys in keys_dont_override, as we anyhow will not override them.
    missing_keys_check_for = [
        key
        for key in config_default_keys
        if key not in set(list(missing_keys_skip) + list(keys_dont_override))
    ]
    missing_keys: List[str] = [
        key
        for key in set(missing_keys_check_for + required_keys)
        if key not in list(config_override.keys())
    ]

    if len(missing_keys) > 0:
        raise ValueError(f"(Missing keys in config: {missing_keys}")

    # We also search for unepxected keys in the pretrain config, but we do
    # not raise an error, leaving the responsibility to the user to assert
    # if the config is correct.
    unexpected_keys: List[str] = [
        key
        for key in config_override.keys()
        if key not in config_default_keys + ["PT", "BiSSL"]
    ]
    if len(unexpected_keys) > 0:
        print(f"(Unexpected keys in config: {unexpected_keys}). They are skipped.")

    # Remove unexpected keys from the config_override dict, as this would otherwise
    # cause the unexpected keys to be added to the config, which is not desired behavior.

    assert all(key in config_default_keys for key in keys_dont_override), (
        f"Keys in keys_dont_override must be in the pretrain config. "
        f"Received keys_dont_override: {keys_dont_override}, "
        f"Expected keys_dont_override to be a subset of: {config_default_keys}"
    )
    for key in set(unexpected_keys + keys_dont_override):  # type: ignore
        config_override.pop(key, None)

    config_override.pop("PT", None)
    config_override.pop("BiSSL", None)

    # Update the config with the config_override dict.
    config.from_dict(config_override)
