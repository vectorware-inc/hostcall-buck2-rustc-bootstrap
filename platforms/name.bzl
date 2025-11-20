load(
    "@prelude//cfg/modifier:cfg_constructor.bzl",
    "PostConstraintAnalysisParams",
    prelude_post_constraint_analysis = "cfg_constructor_post_constraint_analysis",
)
load("@prelude//platforms:defs.bzl", "host_configuration")

def platform_info_label(constraints: dict[TargetLabel, ConstraintValueInfo]) -> str:
    settings = {}
    for constraint in constraints.values():
        settings[str(constraint.setting.label)] = constraint.label.name

    stage = settings.get("rust//constraints:bootstrap-stage")
    workspace = settings.get("rust//constraints:workspace")
    build_script = settings.get("rust//constraints:build-script")
    os = settings.get("prelude//os/constraints:os")
    cpu = settings.get("prelude//cpu/constraints:cpu")
    target = settings.get("rust//constraints:target")

    host_os = host_configuration.os.split(":")[1]
    host_cpu = host_configuration.cpu.split(":")[1]

    target_suffix = ""
    if target and target != "target=host":
        target_suffix = target.split("target=")[1]

    if target == "target=nvptx64" and stage and workspace:
        if build_script == "build-script=true":
            return "rust//platforms/nvptx:{}-build-script".format(workspace)
        return "rust//platforms/nvptx:{}".format(workspace)

    if not stage and not workspace and not build_script and os == host_os and cpu == host_cpu:
        return "rust//platforms:host"

    if stage and workspace and build_script:
        label = "rust//platforms/{}:{}".format(stage, workspace)
        if build_script == "build-script=true":
            label += "-build-script"
        if target_suffix:
            label += "-{}".format(target_suffix)
        return label

    if os and cpu:
        base = "{}-{}".format(os, cpu)
        if target_suffix:
            base = "{}-{}".format(base, target_suffix)
        return base

    if os or cpu:
        base = os or cpu
        if target_suffix:
            base = "{}-{}".format(base, target_suffix)
        return base

    if len(settings) == 0:
        return "null"

    return "cfg"

def cfg_constructor_post_constraint_analysis(
        *,
        refs: dict[str, ProviderCollection],
        params: PostConstraintAnalysisParams) -> PlatformInfo:
    platform = prelude_post_constraint_analysis(refs = refs, params = params)
    return PlatformInfo(
        label = platform_info_label(platform.configuration.constraints),
        configuration = platform.configuration,
    )
