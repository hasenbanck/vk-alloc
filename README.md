# vk-alloc

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

A collection of Vulkan memory allocators written in Rust.

Currently targets [ash](https://github.com/MaikKlein/ash).

## Status

Crate has test cases, but hasn't been tested in production yet. Consider this crate unstable for now.

## Features

All features are optional by default.

* `tracing` Adds logging using [tracing](https://github.com/tokio-rs/tracing).
* `profiling` Adds support for [profiling](https://github.com/aclysma/profiling).
* `vk-buffer-device-address`: Enables the usage of "vkGetBufferDeviceAddress". Either needs the
  "VK_KHR_buffer_device_address" extension loaded or the "bufferDeviceAddress" device feature enabled.
* `vk-dedicated-allocation`: Activates helper functions that decide if an allocation should get it's own dedicated
  memory block. Needs the "VK_KHR_dedicated_allocation" and "VK_KHR_get_memory_requirements2" extensions enabled.

## License

Licensed under MIT or Apache-2.0.
