# 🌋 Vulkan Starter App
## Getting started

You need C++ compiler, Vulkan SDK and CMake installed before you can build this project.

This project uses C++20 standard and thus requires either of those compilers:
- GCC 10.X
- Clang 10
- Microsoft Visual Studio 2019

This is officially tested on *Windows* and *GNU/Linux platforms*, no *macOS* support yet.

## Downloading the repository
Start by cloning the repository with `git clone --depth 1 https://github.com/vladeemerr/vulkan-starter-app`

This repository does not contain any submodules, it utilizes CMake's `FetchContent` feature instead.

## Configuring the project
Run either:
```bash
cmake --preset debug
cmake --preset msvc-debug
cmake --preset mingw-debug
```

## Building
```bash
cmake --build build-debug --parallel
```

### Running
Make sure your working directory is the project root.
