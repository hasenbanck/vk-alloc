# vk-alloc

[![Latest version](https://img.shields.io/crates/v/vk-alloc.svg)](https://crates.io/crates/vk-alloc)
[![Documentation](https://docs.rs/vk-alloc/badge.svg)](https://docs.rs/vk-alloc)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

A segregated list memory allocator for Vulkan written in Rust.

Currently targets Vulkan 1.2+ using [ash](https://github.com/MaikKlein/ash).

## Status

Crate has test cases, but hasn't been tested in production yet. Consider this crate unstable for now.

## Features

All features are optional by default.

* `tracing` Adds logging using [tracing](https://github.com/tokio-rs/tracing).
* `profiling` Adds support for [profiling](https://github.com/aclysma/profiling).
* `vk-buffer-device-address`: Enables the usage of "vkGetBufferDeviceAddress". Either needs the
  "VK_KHR_buffer_device_address" extension loaded or the "bufferDeviceAddress" device feature enabled.

## License

Licensed under MIT or Apache-2.0.
