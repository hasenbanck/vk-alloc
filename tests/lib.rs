use ash::vk;

use vk_alloc::{GeneralAllocator, GeneralAllocatorDescriptor};

pub mod fixture;

#[test]
fn vulkan_context_creation() {
    fixture::VulkanContext::new(vk::make_version(1, 0, 0));
}

#[test]
fn global_allocator_creation() {
    let ctx = fixture::VulkanContext::new(vk::make_version(1, 0, 0));
    GeneralAllocator::new(
        &ctx.instance,
        ctx.physical_device,
        &ctx.logical_device,
        &GeneralAllocatorDescriptor {
            ..Default::default()
        },
    );
}
