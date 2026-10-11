fn main() {
    #[cfg(feature = "runtime-package")]
    {
        println!("cargo:rerun-if-changed=proto/runtime_package.proto");
        let mut config = prost_build::Config::new();
        config
            .compile_protos(&["proto/runtime_package.proto"], &["proto"])
            .expect("compile runtime package protobuf schema");
    }
}
