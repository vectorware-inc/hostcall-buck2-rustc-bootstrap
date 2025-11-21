use crate::spec::{
    LinkSelfContainedDefault, LinkerFlavor, MergeFunctions, PanicStrategy, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        arch: "nvptx64".into(),
        data_layout: "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64".into(),
        llvm_target: "nvptx64-nvidia-cuda".into(),
        metadata: TargetMetadata {
            description: Some("--emit=asm generates PTX code that runs on NVIDIA GPUs (VectorWare)".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        options: TargetOptions {
            os: "cuda".into(),
            env: "unknown".into(),
            vendor: "vectorware".into(),
            linker_flavor: LinkerFlavor::Ptx,
            linker: Some("rust-ptx-linker".into()),
            cpu: "sm_89".into(),
            max_atomic_width: Some(64),
            panic_strategy: PanicStrategy::Abort,
            only_cdylib: true,
            obj_is_bitcode: true,
            dll_prefix: "".into(),
            dll_suffix: ".ptx".into(),
            exe_suffix: ".ptx".into(),
            merge_functions: MergeFunctions::Disabled,
            supports_stack_protector: false,
            link_self_contained: LinkSelfContainedDefault::True,
            ..Default::default()
        },
    }
}
