Plan for moving `hostcall-demo` fully onto buck2-rustc-bootstrap.

- [x] Inspect current state: host target exists via `rust_bootstrap_*` rules; kernel is still built with a `genrule` shelling out to Cargo; libc fixup for `nvptx64` points at `//hostcall-demo:libc_hostcall`; nvptx platform/target spec present.
- [x] Define buck-native kernel target that builds PTX with the bootstrap toolchain (no external `build.sh`), reusing/updating the nvptx platform/target spec.
- [x] Ensure libc swap for PTX is wired correctly via fixups (and any needed std/alloc tweaks) so the kernel links against `libc-hostcall`.
- [x] Hook host binary to depend on the buck-built PTX artifact (no env/manual pathing), replacing the current `genrule` plumbing.
- [ ] Clean up legacy scripts/docs (`culinux/build.sh`, README/env instructions) to reflect the buck-first workflow and add any helper targets if needed.

Notes:
- Added and renamed the nvptx target spec to `nvptx64-vectorware-linux` and started plumbing target metadata into platform labels to unblock nvptx builds.
- Built `libc_hostcall_nvptx` successfully by using `_rust_toolchain=toolchains//:rust_nvptx` and the nvptx platform; nvptx sysroot builds (`stage1:sysroot --target-platforms //platforms/nvptx:library`) now succeed. Need to add the buck-native PTX kernel target and host wiring next.
