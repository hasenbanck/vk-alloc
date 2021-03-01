# vk-alloc

[![Latest version](https://img.shields.io/crates/v/vk-alloc.svg)](https://crates.io/crates/vk-alloc)
[![Documentation](https://docs.rs/vk-alloc/badge.svg)](https://docs.rs/vk-alloc)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

A segregated list memory allocator for Vulkan written in Rust.

Targets Vulkan 1.2+ using [erupt](https://gitlab.com/Friz64/erupt).

## Status

Crate has test cases, but hasn't been tested in production yet. Consider this crate unstable for now.

## Features

All features are optional by default.

* `tracing` Adds logging using [tracing](https://github.com/tokio-rs/tracing).
* `profiling` Adds support for [profiling](https://github.com/aclysma/profiling).
* `vk-buffer-device-address`: Enables the usage of "vkGetBufferDeviceAddress". Either needs the the
  Vulkan 1.2 feature "bufferDeviceAddress" device feature enabled.

## Older versions

Up until version 0.3.0 this allocator supported ash. With 0.4.0 I switched to erupt. If you want to
continue using ash, please stay on the 0.3.0 release. It might get minor bugfixes if needed.

## License

Licensed under MIT or Apache-2.0.
