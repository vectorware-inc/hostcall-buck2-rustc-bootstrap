#!/bin/bash
set -e
RUSTFLAGS="--cfg target_has_reliable_f128=false -C panic=abort -C linker-flavor=llbc -Z unstable-options -C target-feature=-crt-static -C link-args=-nostdlib\ -nostartfiles\ -nodefaultlibs"
cargo build --release --target nvptx64-nvidia-linux.json -Zbuild-std="core,std,panic_abort" -vvv
rm target/objects/ -rf
mkdir -p target/objects
cd target/objects
ar x ../nvptx64-nvidia-linux/release/libculinux.a
cd ../..
echo "Linking..."
llvm-link target/objects/*.rcgu.o -o target/libculinux.o
llvm-dis target/libculinux.o -o target/libculinux.ll
echo "DCE..."
opt -passes='internalize,globaldce'  target/libculinux.o  -o target/libculinux.opt.o
echo "Optimizing..."
opt -O1  target/libculinux.opt.o  -o target/libculinux.opt.o
echo "Compiling..."
llc -mcpu=sm_89 target/libculinux.opt.o -o target/libculinux.ptx
echo "PTX fixup..."
sed -i 's/sm_75/sm_89/g' target/libculinux.ptx
cp target/libculinux.ptx test.ptx
echo "Starting up the hostcall runtime.."
../target/release/hostcalls