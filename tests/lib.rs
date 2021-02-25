use ash::vk;

use vk_alloc::{
    Allocation, AllocationDescriptor, AllocationType, Allocator, AllocatorDescriptor,
    MemoryLocation,
};

pub mod fixture;

#[test]
fn vulkan_context_creation() {
    fixture::VulkanContext::new(vk::make_version(1, 2, 0));
}

#[test]
fn allocator_creation() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
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
fn allocator_simple_free() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MiB
    );

    let allocation = alloc
        .allocate(&AllocationDescriptor {
            location: MemoryLocation::GpuOnly,
            requirements: vk::MemoryRequirements::builder()
                .alignment(512)
                .size(1024)
                .memory_type_bits(u32::MAX)
                .build(),
            allocation_type: AllocationType::Buffer,
            is_dedicated: false,
        })
        .unwrap();

    assert_eq!(allocation.size, 1024);
    assert_eq!(allocation.offset, 0);

    alloc.free(&allocation).unwrap();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn allocator_allocation_1024() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MiB
    );

    let mut allocations: Vec<Allocation> = (0..1024)
        .into_iter()
        .map(|i| {
            let allocation = alloc
                .allocate(&AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirements::builder()
                        .alignment(512)
                        .size(1024)
                        .memory_type_bits(u32::MAX)
                        .build(),
                    allocation_type: AllocationType::Buffer,
                    is_dedicated: false,
                })
                .unwrap();
            assert_eq!(allocation.size, 1024);
            assert_eq!(allocation.offset, i * 1024);

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
            alloc.free(&allocation).unwrap();

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
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MiB
    );

    let mut allocations: Vec<Allocation> = (0..1024)
        .into_iter()
        .map(|i| {
            let allocation = alloc
                .allocate(&AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirements::builder()
                        .alignment(1024)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build(),
                    allocation_type: AllocationType::Buffer,
                    is_dedicated: false,
                })
                .unwrap();
            assert_eq!(allocation.size, 256);
            assert_eq!(allocation.offset, i * 1024);

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
            alloc.free(&allocation).unwrap();

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

#[test]
fn allocator_reverse_free() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MiB
    );

    let mut allocations: Vec<Allocation> = (0..1024)
        .into_iter()
        .map(|i| {
            let allocation = alloc
                .allocate(&AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirements::builder()
                        .alignment(1024)
                        .size(256)
                        .memory_type_bits(u32::MAX)
                        .build(),
                    allocation_type: AllocationType::Buffer,
                    is_dedicated: false,
                })
                .unwrap();
            assert_eq!(allocation.size, 256);
            assert_eq!(allocation.offset, i * 1024);

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
            assert_eq!(allocation.offset, ((1024 * 1024) - ((i + 1) * 1024)) as u64);

            alloc.free(&allocation).unwrap();

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

#[test]
fn allocator_free_every_second_time() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MiB
    );

    let allocations: Vec<Allocation> = (0..1024)
        .into_iter()
        .map(|_| {
            let allocation = alloc
                .allocate(&AllocationDescriptor {
                    location: MemoryLocation::GpuOnly,
                    requirements: vk::MemoryRequirements::builder()
                        .alignment(1024)
                        .size(1024)
                        .memory_type_bits(u32::MAX)
                        .build(),
                    allocation_type: AllocationType::Buffer,
                    is_dedicated: false,
                })
                .unwrap();
            allocation
        })
        .collect();

    let mut odd: Vec<Allocation> = allocations
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 2 == 0)
        .map(|(_, allocation)| allocation.clone())
        .collect();
    let mut even: Vec<Allocation> = allocations
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 2 == 1)
        .map(|(_, allocation)| allocation.clone())
        .collect();

    odd.drain(..).for_each(|allocation| {
        alloc.free(&allocation).unwrap();
    });

    even.drain(..).for_each(|allocation| {
        alloc.free(&allocation).unwrap();
    });

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}

#[test]
fn allocator_allocation_dedicated() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 2, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MiB
    );

    let allocation = alloc
        .allocate(&AllocationDescriptor {
            location: MemoryLocation::GpuOnly,
            requirements: vk::MemoryRequirements::builder()
                .alignment(512)
                .size(10 * 1024 * 1024) // 10 MiB
                .memory_type_bits(u32::MAX)
                .build(),
            allocation_type: AllocationType::Buffer,
            is_dedicated: false,
        })
        .unwrap();
    assert_eq!(allocation.size, 10 * 1024 * 1024);
    assert_eq!(allocation.offset, 0);

    assert_eq!(alloc.allocation_count(), 1);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 10 * 1024 * 1024);
    assert_eq!(alloc.unused_bytes(), 0);

    alloc.free(&allocation).unwrap();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
}
