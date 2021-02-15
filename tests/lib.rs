use ash::vk;

use vk_alloc::{
    Allocation, AllocationDescriptor, AllocationInfo, AllocationType, Allocator,
    AllocatorDescriptor, AllocatorError, AllocatorStatistic, LinearAllocationDescriptor,
    LinearAllocator, LinearAllocatorDescriptor, MemoryUsage,
};

pub mod fixture;

#[test]
fn vulkan_context_creation() {
    fixture::VulkanContext::new(vk::make_version(1, 0, 0));
}

#[test]
fn allocator_creation() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor {
            ..Default::default()
        },
    );
}

#[test]
fn linear_allocator_creation() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn linear_allocator_allocation_1024() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )
    .unwrap();

    for i in 0..1024 {
        let allocation = alloc
            .allocate(&LinearAllocationDescriptor {
                size: 1024,
                alignment: 512,
                allocation_type: AllocationType::Buffer,
            })
            .unwrap();
        assert_eq!(allocation.size(), 1024);
        assert_eq!(allocation.offset(), i * 1024);
    }

    assert_eq!(alloc.allocation_count(), 1024);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 1024 * 1024);
    assert_eq!(alloc.unused_bytes(), 0);

    alloc.free();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn linear_allocator_allocation_256() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )
    .unwrap();

    for i in 0..1024 {
        let allocation = alloc
            .allocate(&LinearAllocationDescriptor {
                size: 256,
                alignment: 1024,
                allocation_type: AllocationType::Buffer,
            })
            .unwrap();
        assert_eq!(allocation.size(), 256);
        assert_eq!(allocation.offset(), i * 1024);
    }

    assert_eq!(alloc.allocation_count(), 1024);
    assert_eq!(alloc.unused_range_count(), 1023);
    assert_eq!(alloc.used_bytes(), 1024 * 256);
    assert_eq!(alloc.unused_bytes(), 1023 * 768);

    alloc.free();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn linear_allocator_allocation_granularity() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )
    .unwrap();

    let allocation = alloc
        .allocate(&LinearAllocationDescriptor {
            size: 256,
            alignment: 256,
            allocation_type: AllocationType::Buffer,
        })
        .unwrap();
    assert_eq!(allocation.size(), 256);
    assert_eq!(allocation.offset(), 0);

    let allocation = alloc
        .allocate(&LinearAllocationDescriptor {
            size: 256,
            alignment: 256,
            allocation_type: AllocationType::OptimalImage,
        })
        .unwrap();
    assert_eq!(allocation.size(), 256);
    assert_eq!(allocation.offset(), ctx.buffer_image_granularity);

    assert_eq!(alloc.allocation_count(), 2);
    assert_eq!(alloc.unused_range_count(), 1);
    assert_eq!(alloc.used_bytes(), 2 * 256);
    assert_eq!(alloc.unused_bytes(), 768);

    alloc.free();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn linear_allocator_allocation_oom() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )
    .unwrap();

    let allocation = alloc.allocate(&LinearAllocationDescriptor {
        size: 1050000,
        alignment: 256,
        allocation_type: AllocationType::Buffer,
    });

    assert!(allocation.is_err());
    assert_eq!(AllocatorError::OutOfMemory, allocation.err().unwrap());
}

#[test]
fn allocator_simple_free() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MB
    );

    let allocation = alloc
        .allocate(&AllocationDescriptor {
            location: MemoryUsage::GpuOnly,
            requirements: vk::MemoryRequirements::builder()
                .alignment(512)
                .size(1024)
                .memory_type_bits(u32::MAX)
                .build(),
            allocation_type: AllocationType::Buffer,
            is_dedicated: false,
        })
        .unwrap();

    assert_eq!(allocation.size(), 1024);
    assert_eq!(allocation.offset(), 0);

    alloc.free(allocation).unwrap();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn allocator_allocation_1024() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MB
    );

    let mut allocations: Vec<Allocation> = (0..1024)
        .into_iter()
        .map(|i| {
            let allocation = alloc
                .allocate(&AllocationDescriptor {
                    location: MemoryUsage::GpuOnly,
                    requirements: vk::MemoryRequirements::builder()
                        .alignment(512)
                        .size(1024)
                        .memory_type_bits(u32::MAX)
                        .build(),
                    allocation_type: AllocationType::Buffer,
                    is_dedicated: false,
                })
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
            alloc.free(allocation).unwrap();

            assert_eq!(alloc.allocation_count(), (1023 - i));
            assert_eq!(alloc.used_bytes(), 1024 * (1023 - i) as u64);
        });

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn allocator_allocation_256() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MB
    );

    let mut allocations: Vec<Allocation> = (0..1024)
        .into_iter()
        .map(|i| {
            let allocation = alloc
                .allocate(&AllocationDescriptor {
                    location: MemoryUsage::GpuOnly,
                    requirements: vk::MemoryRequirements::builder()
                        .alignment(1024)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build(),
                    allocation_type: AllocationType::Buffer,
                    is_dedicated: false,
                })
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
            alloc.free(allocation).unwrap();

            assert_eq!(alloc.allocation_count(), 1023 - i);
            assert_eq!(alloc.used_bytes(), 256 * (1023 - i) as u64);
            // sic! We free from the front. The padding is part of the next chunk.
            assert_eq!(alloc.unused_range_count(), 1023 - i);
            assert_eq!(alloc.unused_bytes(), 768 * (1023 - i) as u64);
        });

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}
