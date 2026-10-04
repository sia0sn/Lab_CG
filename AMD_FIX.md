# AMD/Vulkan starter fix

This copy is based on the current `master` of `vladeemerr/vulkan-starter-app`.

Modified:
- `source/graphics_internal.hpp`
- `source/graphics_internal.cpp`

The rest of the project structure is preserved.

The fix adds explicit presentation-queue handling, checks vk-bootstrap results before `.value()`,
uses the presentation queue for `vkQueuePresentKHR`, and makes swapchain rebuild failure-safe.

Build:
    cmake --preset debug
    cmake --build build-debug --parallel

Run the executable with the project root as the working directory.
