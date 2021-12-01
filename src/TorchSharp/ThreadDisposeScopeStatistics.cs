#nullable enable
namespace TorchSharp
{
    /// <summary>
    /// Keeps track of statistics for a dispose scope. Can be queried to figure out performance/memory issues.
    /// </summary>
    public class ThreadDisposeScopeStatistics
    {
        /// <summary>
        /// The number of disposables that were created on this thread, but weren't captured by a DisposeScope.
        /// </summary>
        public long CreatedOutsideScopeCount { get; internal set; }

        /// <summary>
        /// The number of disposables that were created on this thread and were captured by a DisposeScope.
        /// </summary>
        public long CreatedInScopeCount { get; internal set; }

        /// <summary>
        /// The number of disposables that were disposed on this thread and were disposed while in a DisposeScope.
        /// </summary>
        public long DisposedInScopeCount { get; internal set; }

        /// <summary>
        /// Number of disposables that were once included in the scope, but were subsequently detached.
        /// </summary>
        public long DetachedFromScopeCount { get; internal set; }

        /// <summary>
        /// The number of disposables that are currently live on the current thread. If a It's aproximate, see
        /// Tensor.TotalCount. Disposables that weren't created within a DisposeScope, or detached from the dispose
        /// scope, will not be counted as alive.
        /// </summary>
        public long ThreadTotalLiveCount => CreatedInScopeCount - DisposedInScopeCount - DetachedFromScopeCount;

        /// <summary>
        /// Resets the counts for the current thread. See ThreadTotalLiveCount etc. Mainly used in tests to make sure
        /// we get a clean slate on the thread.
        /// </summary>
        public void Reset()
        {
            CreatedOutsideScopeCount = 0;
            CreatedInScopeCount = 0;
            DisposedInScopeCount = 0;
            DetachedFromScopeCount = 0;
        }
    }
}