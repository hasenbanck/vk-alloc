use ash::vk;

use vk_alloc::{
    AllocationDescriptor, AllocationInfo, AllocationType, Allocator, AllocatorDescriptor,
    AllocatorError, AllocatorStatistic, LinearAllocationDescriptor, LinearAllocator,
    LinearAllocatorDescriptor, MemoryUsage,
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
fn linear_allocator_creation() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            ..Default::default()
        },
    )?;
    Ok(())
}

#[test]
fn linear_allocator_allocation_1024() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )?;

    for i in 0..1024 {
        let allocation = alloc.allocate(&LinearAllocationDescriptor {
            size: 1024,
            alignment: 512,
            allocation_type: AllocationType::Buffer,
        })?;
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

    Ok(())
}

#[test]
fn linear_allocator_allocation_256() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )?;

    for i in 0..1024 {
        let allocation = alloc.allocate(&LinearAllocationDescriptor {
            size: 256,
            alignment: 1024,
            allocation_type: AllocationType::Buffer,
        })?;
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

    Ok(())
}

#[test]
fn linear_allocator_allocation_granularity() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )?;

    let allocation = alloc.allocate(&LinearAllocationDescriptor {
        size: 256,
        alignment: 256,
        allocation_type: AllocationType::Buffer,
    })?;
    assert_eq!(allocation.size(), 256);
    assert_eq!(allocation.offset(), 0);

    let allocation = alloc.allocate(&LinearAllocationDescriptor {
        size: 256,
        alignment: 256,
        allocation_type: AllocationType::OptimalImage,
    })?;
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

    Ok(())
}

#[test]
fn linear_allocator_allocation_oom() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = LinearAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &LinearAllocatorDescriptor {
            location: MemoryUsage::CpuToGpu,
            block_size: 20, // 1 MB
        },
    )?;

    let allocation = alloc.allocate(&LinearAllocationDescriptor {
        size: 1050000,
        alignment: 256,
        allocation_type: AllocationType::Buffer,
    });

    assert!(allocation.is_err());
    assert_eq!(AllocatorError::OutOfMemory, allocation.err().unwrap());

    Ok(())
}

#[test]
fn allocator_allocation_1024() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MB
    );

    for i in 0..1024 {
        let allocation = alloc.allocate(&AllocationDescriptor {
            location: MemoryUsage::GpuOnly,
            requirements: vk::MemoryRequirements::builder()
                .alignment(512)
                .size(1024)
                .memory_type_bits(u32::MAX)
                .build(),
            allocation_type: AllocationType::Buffer,
            is_dedicated: false,
        })?;
        assert_eq!(allocation.size(), 1024);
        assert_eq!(allocation.offset(), i * 1024);
    }

    assert_eq!(alloc.allocation_count(), 1024);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 1024 * 1024);
    assert_eq!(alloc.unused_bytes(), 0);

    // TODO
    /*
    alloc.free();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
    */

    Ok(())
}

#[test]
fn allocator_allocation_256() -> Result<(), AllocatorError> {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    let mut alloc = Allocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &AllocatorDescriptor { block_size: 20 }, // 1 MB
    );

    for i in 0..1024 {
        let allocation = alloc.allocate(&AllocationDescriptor {
            location: MemoryUsage::GpuOnly,
            requirements: vk::MemoryRequirements::builder()
                .alignment(1024)
                .size(256)
                .memory_type_bits(u32::MAX)
                .build(),
            allocation_type: AllocationType::Buffer,
            is_dedicated: false,
        })?;
        assert_eq!(allocation.size(), 256);
        assert_eq!(allocation.offset(), i * 1024);
    }

    assert_eq!(alloc.allocation_count(), 1024);
    assert_eq!(alloc.unused_range_count(), 1023);
    assert_eq!(alloc.used_bytes(), 1024 * 256);
    assert_eq!(alloc.unused_bytes(), 1023 * 768);

    // TODO
    /*
    alloc.free();

    assert_eq!(alloc.allocation_count(), 0);
    assert_eq!(alloc.unused_range_count(), 0);
    assert_eq!(alloc.used_bytes(), 0);
    assert_eq!(alloc.unused_bytes(), 0);
    */

    Ok(())
}
