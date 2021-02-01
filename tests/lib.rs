use ash::vk;

pub mod fixture;

#[test]
fn vulkan_context_creation() {
    #[cfg(feature = "tracing")]
        tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    fixture::VulkanContext::new(vk::make_version(1, 0, 0));
}
