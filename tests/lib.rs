use erupt::vk;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use vk_alloc::{Allocation, AllocationDescriptor, Allocator, AllocatorDescriptor, MemoryLocation};

pub mod fixture;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
enum TestLifetime {
    Static,
}

impl vk_alloc::Lifetime for TestLifetime {}

#[test]
fn vulkan_context_creation() {
    fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
}

#[test]
fn allocator_creation() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        Allocator::<TestLifetime>::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor {
                ..Default::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn allocator_simple_free() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let allocation = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(512)
                        .size(1024)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();

        assert_eq!(allocation.size(), 1024);
        assert_eq!(allocation.offset(), 0);

        alloc.deallocate(&ctx.logical_device, &allocation).unwrap();

        assert_eq!(alloc.allocation_count(), 0);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_allocation_1024() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let mut allocations: Vec<Allocation<_>> = (0..1024)
            .into_iter()
            .map(|i| {
                let allocation = alloc
                    .allocate(
                        &ctx.logical_device,
                        &AllocationDescriptor {
                            location: MemoryLocation::GpuOnly,
                            requirements: vk::MemoryRequirementsBuilder::new()
                                .alignment(512)
                                .size(1024)
                                .memory_type_bits(u32::MAX)
                                .build_dangling(),
                            lifetime: TestLifetime::Static,
                            is_dedicated: false,
                            is_optimal: false,
                        },
                    )
                    .unwrap();
                assert_eq!(allocation.size(), 1024);
                assert_eq!(allocation.offset(), i * 1024);

                allocation
            })
            .collect();

        assert_eq!(alloc.allocation_count(), 1024);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 1024 * 1024);
        assert_eq!(alloc.unused_bytes(), 0);

        allocations
            .drain(..)
            .enumerate()
            .for_each(|(i, allocation)| {
                alloc.deallocate(&ctx.logical_device, &allocation).unwrap();

                assert_eq!(alloc.allocation_count(), (1023 - i));
                assert_eq!(alloc.used_bytes(), 1024 * (1023 - i) as vk::DeviceSize);
            });

        assert_eq!(alloc.allocation_count(), 0);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_allocation_256() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let mut allocations: Vec<Allocation<_>> = (0..1024)
            .into_iter()
            .map(|i| {
                let allocation = alloc
                    .allocate(
                        &ctx.logical_device,
                        &AllocationDescriptor {
                            location: MemoryLocation::GpuOnly,
                            requirements: vk::MemoryRequirementsBuilder::new()
                                .alignment(1024)
                                .size(256)
                                .memory_type_bits(u32::MAX)
                                .build_dangling(),
                            lifetime: TestLifetime::Static,
                            is_dedicated: false,
                            is_optimal: false,
                        },
                    )
                    .unwrap();
                assert_eq!(allocation.size(), 256);
                assert_eq!(allocation.offset(), i * 1024);

                allocation
            })
            .collect();

        assert_eq!(alloc.allocation_count(), 1024);
        assert_eq!(alloc.unused_range_count(), 1023);
        assert_eq!(alloc.used_bytes(), 256 * 1024);
        assert_eq!(alloc.unused_bytes(), 768 * 1023);

        allocations
            .drain(..)
            .enumerate()
            .for_each(|(i, allocation)| {
                alloc.deallocate(&ctx.logical_device, &allocation).unwrap();

                assert_eq!(alloc.allocation_count(), 1023 - i);
                assert_eq!(alloc.used_bytes(), 256 * (1023 - i) as vk::DeviceSize);
                // sic! We free from the front. The padding is part of the next chunk.
                assert_eq!(alloc.unused_range_count(), 1023 - i);
                assert_eq!(alloc.unused_bytes(), 768 * (1023 - i) as vk::DeviceSize);
            });

        assert_eq!(alloc.allocation_count(), 0);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_reverse_free() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let mut allocations: Vec<Allocation<_>> = (0..1024)
            .into_iter()
            .map(|i| {
                let allocation = alloc
                    .allocate(
                        &ctx.logical_device,
                        &AllocationDescriptor {
                            location: MemoryLocation::GpuOnly,
                            requirements: vk::MemoryRequirementsBuilder::new()
                                .alignment(1024)
                                .size(256)
                                .memory_type_bits(u32::MAX)
                                .build_dangling(),
                            lifetime: TestLifetime::Static,
                            is_dedicated: false,
                            is_optimal: false,
                        },
                    )
                    .unwrap();
                assert_eq!(allocation.size(), 256);
                assert_eq!(allocation.offset(), i * 1024);

                allocation
            })
            .collect();

        assert_eq!(alloc.allocation_count(), 1024);
        assert_eq!(alloc.unused_range_count(), 1023);
        assert_eq!(alloc.used_bytes(), 256 * 1024);
        assert_eq!(alloc.unused_bytes(), 768 * 1023);

        allocations
            .drain(..)
            .rev()
            .enumerate()
            .for_each(|(i, allocation)| {
                assert_eq!(
                    allocation.offset(),
                    ((1024 * 1024) - ((i + 1) * 1024)) as vk::DeviceSize
                );

                alloc.deallocate(&ctx.logical_device, &allocation).unwrap();

                assert_eq!(alloc.allocation_count(), 1023 - i);
                assert_eq!(alloc.used_bytes(), 256 * (1023 - i) as vk::DeviceSize);

                // the last part not in use is a free chunk!
                if i < 1023 {
                    assert_eq!(alloc.unused_range_count(), 1022 - i);
                    assert_eq!(alloc.unused_bytes(), 768 * (1022 - i) as vk::DeviceSize);
                }
            });

        assert_eq!(alloc.allocation_count(), 0);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_free_every_second_time() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let allocations: Vec<Allocation<_>> = (0..1024)
            .into_iter()
            .map(|_| {
                let allocation = alloc
                    .allocate(
                        &ctx.logical_device,
                        &AllocationDescriptor {
                            location: MemoryLocation::GpuOnly,
                            requirements: vk::MemoryRequirementsBuilder::new()
                                .alignment(1024)
                                .size(1024)
                                .memory_type_bits(u32::MAX)
                                .build_dangling(),
                            lifetime: TestLifetime::Static,
                            is_dedicated: false,
                            is_optimal: false,
                        },
                    )
                    .unwrap();
                allocation
            })
            .collect();

        let mut odd: Vec<Allocation<_>> = allocations
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 0)
            .map(|(_, allocation)| allocation.clone())
            .collect();
        let mut even: Vec<Allocation<_>> = allocations
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 1)
            .map(|(_, allocation)| allocation.clone())
            .collect();

        odd.drain(..).for_each(|allocation| {
            alloc.deallocate(&ctx.logical_device, &allocation).unwrap();
        });

        even.drain(..).for_each(|allocation| {
            alloc.deallocate(&ctx.logical_device, &allocation).unwrap();
        });

        assert_eq!(alloc.allocation_count(), 0);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_allocation_dedicated() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let allocation = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(512)
                        .size(10 * 1024 * 1024) // 10 MiB
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();
        assert_eq!(allocation.size(), 10 * 1024 * 1024);
        assert_eq!(allocation.offset(), 0);

        assert_eq!(alloc.allocation_count(), 1);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 10 * 1024 * 1024);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.deallocate(&ctx.logical_device, &allocation).unwrap();

        assert_eq!(alloc.allocation_count(), 0);
        assert_eq!(alloc.unused_range_count(), 0);
        assert_eq!(alloc.used_bytes(), 0);
        assert_eq!(alloc.unused_bytes(), 0);

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_properly_merge_free_entries() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let a0 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();
        let a1 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();
        let a2 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();
        let a3 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();

        alloc.deallocate(&ctx.logical_device, &a3).unwrap();
        alloc.deallocate(&ctx.logical_device, &a1).unwrap();
        alloc.deallocate(&ctx.logical_device, &a2).unwrap();
        alloc.deallocate(&ctx.logical_device, &a0).unwrap();

        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_fuzzy() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor::default(),
        )
        .unwrap();

        let mut allocations: Vec<(u8, Allocation<_>)> = Vec::with_capacity(10_0000);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);

        for i in 0..10_000 {
            let is_allocation = rng.gen_bool(0.6);

            if (is_allocation && !allocations.is_empty()) || allocations.is_empty() {
                let value: u8 = rng.gen_range(0..=255);
                let size = (value as usize + 1) * 4;

                let mut allocation = alloc
                    .allocate(
                        &ctx.logical_device,
                        &AllocationDescriptor {
                            location: MemoryLocation::CpuToGpu,
                            requirements: vk::MemoryRequirementsBuilder::new()
                                .alignment(256)
                                .size(size as u64)
                                .memory_type_bits(u32::MAX)
                                .build_dangling(),
                            lifetime: TestLifetime::Static,
                            is_dedicated: false,
                            is_optimal: false,
                        },
                    )
                    .unwrap();
                assert!(size <= allocation.size() as usize);

                let slice = allocation.mapped_slice_mut().unwrap().unwrap();
                slice.fill(value);
                allocations.push((value, allocation));
            } else {
                let select: usize = rng.gen_range(0..allocations.len());
                let (value, allocation) = allocations.remove(select);
                let slice = allocation.mapped_slice().unwrap().unwrap();

                slice.iter().for_each(|x| {
                    if *x != value {
                        panic!("After {} iters: {} != {}: {:?}", i, *x, value, slice);
                    }
                });

                alloc.deallocate(&ctx.logical_device, &allocation).unwrap();
            }
        }
        alloc.cleanup(&ctx.logical_device);
    }
}

#[test]
fn allocator_granularity() {
    unsafe {
        let ctx = fixture::VulkanContext::new(vk::make_api_version(0, 1, 2, 0));
        let alloc = Allocator::new(
            &ctx.instance,
            ctx.physical_device,
            &AllocatorDescriptor { block_size: 20 }, // 1 MiB
        )
        .unwrap();

        let allocation1 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(512)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();

        assert_eq!(allocation1.size(), 512);
        assert_eq!(allocation1.offset(), 0);

        let allocation2 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(1024)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: true,
                },
            )
            .unwrap();

        assert_eq!(allocation2.size(), 1024);
        assert_eq!(allocation2.offset(), 1024);

        alloc.deallocate(&ctx.logical_device, &allocation2).unwrap();

        let allocation3 = alloc
            .allocate(
                &ctx.logical_device,
                &AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirementsBuilder::new()
                        .alignment(256)
                        .size(1024)
                        .memory_type_bits(u32::MAX)
                        .build_dangling(),
                    lifetime: TestLifetime::Static,
                    is_dedicated: false,
                    is_optimal: false,
                },
            )
            .unwrap();

        assert_eq!(allocation3.size(), 1024);
        assert_eq!(allocation3.offset(), 512);

        alloc.cleanup(&ctx.logical_device);
    }
}
