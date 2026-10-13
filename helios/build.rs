fn main() {
    #[cfg(feature = "runtime-package")]
    {
        println!("cargo:rerun-if-changed=proto/runtime_package.proto");
        let mut config = prost_build::Config::new();
        config
            .compile_protos(&["proto/runtime_package.proto"], &["proto"])
            .expect("compile runtime package protobuf schema");
    }
    #[cfg(feature = "roce")]
    {
        println!("cargo:rerun-if-changed=native/roce_proxy.c");
        cc::Build::new()
            .file("native/roce_proxy.c")
            .std("gnu11")
            .compile("flare_roce_proxy");
        println!("cargo:rustc-link-lib=ibverbs");
        println!("cargo:rustc-link-lib=pthread");
    }
}
