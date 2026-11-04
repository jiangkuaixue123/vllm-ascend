#ifndef TILING_ARGS_H
#define TILING_ARGS_H
#include <cstdint>

namespace Moe {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 3U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 204U * 1024UL * 1024UL;
constexpr uint64_t ROUTE_BEFORE_DOWN_ROUTING_KEY = 21000;

constexpr bool IsRouteBeforeDownRoutingSupported(bool enabled, bool tilingSucceeded, uint64_t key)
{
    // The opt-in path only supports full-load dynamic routing. In particular,
    // the inherited gather path does not allocate its per-row scale metadata.
    // Preserve the validation behavior of existing, non-opt-in consumers.
    return !enabled || (tilingSucceeded && key == ROUTE_BEFORE_DOWN_ROUTING_KEY);
}
}  // namespace Moe
#endif  // TILING_ARGS_H
