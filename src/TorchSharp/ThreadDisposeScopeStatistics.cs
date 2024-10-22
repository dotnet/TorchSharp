#nullable enable
using System.Text;
using TorchSharp;

namespace TorchSharp
{
    public interface ILifetimeStatistics
    {
        /// <summary>
        /// The number of disposables that were created on this thread, but weren't captured by a DisposeScope.
        /// </summary>
        long CreatedOutsideScopeCount { get; }
        /// <summary>
        /// The number of disposables that were Disposed on this thread, but weren't captured by a DisposeScope.
        /// </summary>
        long DisposedOutsideScopeCount { get; }
        /// <summary>
        /// The number of disposables that were created on this thread and were captured by a DisposeScope.
        /// </summary>
        long CreatedInScopeCount { get; }
        /// <summary>
        /// The number of disposables that were disposed on this thread and were disposed while in a DisposeScope.
        /// </summary>
        long DisposedInScopeCount { get; }
        /// <summary>
        /// The number of disposables that were created on this thread outside a DisposeScope, and then
        /// eventually were attached to one.
        /// </summary>
        long AttachedToScopeCount { get; }
        /// <summary>
        /// Number of disposables that were once included in the scope, but were subsequently detached.
        /// </summary>
        long DetachedFromScopeCount { get; }
        /// <summary>
        /// The number of disposables that are currently live on the current thread.
        /// </summary>
        long ThreadTotalLiveCount { get; }
        /// <summary>
        /// Resets the counts for the current thread. See ThreadTotalLiveCount etc. Mainly used in tests to make sure
        /// we get a clean slate on the thread.
        /// </summary>
        void Reset();
        /// <summary>
        /// A debug printout of all the properties and their values, suitable for a log or console output
        /// </summary>
        /// <returns></returns>
        string ToString();
    }

    /// <summary>
    /// Keeps track of the combined Tensor and PackedSequence statistics for the current thread. Can be queried to figure out performance/memory issues.
    /// </summary>
    public class ThreadDisposeScopeStatistics: ILifetimeStatistics
    {
        public long CreatedOutsideScopeCount => _TensorStatistics.CreatedOutsideScopeCount +
                                                _PackedSequenceStatistics.CreatedOutsideScopeCount;

        public long DisposedOutsideScopeCount => _TensorStatistics.DisposedOutsideScopeCount +
                                                _PackedSequenceStatistics.DisposedOutsideScopeCount;
        public long CreatedInScopeCount => _TensorStatistics.CreatedInScopeCount +
                                           _PackedSequenceStatistics.CreatedInScopeCount;

        public long DisposedInScopeCount => _TensorStatistics.DisposedInScopeCount +
                                            _PackedSequenceStatistics.DisposedInScopeCount;

        public long AttachedToScopeCount=> _TensorStatistics.AttachedToScopeCount +
                                             _PackedSequenceStatistics.AttachedToScopeCount;

        public long DetachedFromScopeCount=> _TensorStatistics.DetachedFromScopeCount +
                                             _PackedSequenceStatistics.DetachedFromScopeCount;

        public long ThreadTotalLiveCount => _TensorStatistics.ThreadTotalLiveCount +
                                            _PackedSequenceStatistics.ThreadTotalLiveCount;

        public void Reset()
        {
            _TensorStatistics.Reset();
            _PackedSequenceStatistics.Reset();
        }

        public override string ToString()
        {
            return LifetimeStatisticsUtil.DebugString(this);
        }
        /// <summary>
        /// Keeps track of the Tensor statistics for the current thread. Can be queried to figure out performance/memory issues.
        /// </summary>
        public ILifetimeStatistics TensorStatistics => _TensorStatistics;
        internal LifetimeStatistics _TensorStatistics { get; set; } = new LifetimeStatistics();
        /// <summary>
        /// Keeps track of the PackedSequence statistics for the current thread. Can be queried to figure out performance/memory issues.
        /// </summary>
        public ILifetimeStatistics PackedSequenceStatistics => _PackedSequenceStatistics;
        internal LifetimeStatistics _PackedSequenceStatistics { get; set; } = new LifetimeStatistics();
    }

    public class LifetimeStatistics : ILifetimeStatistics
    {
        public long CreatedOutsideScopeCount { get; internal set; }
        public long DisposedOutsideScopeCount { get; internal set; }
        public long CreatedInScopeCount { get; internal set; }
        public long DisposedInScopeCount { get; internal set; }
        public long AttachedToScopeCount { get; internal set; }
        public long DetachedFromScopeCount { get; internal set; }
        /// <summary>
        /// Exact number of objects that are still live. Is the difference of all created objects
        /// minus all disposed objects.
        /// </summary>
        public long ThreadTotalLiveCount => (CreatedInScopeCount - DisposedInScopeCount) + (CreatedOutsideScopeCount - DisposedOutsideScopeCount);

        public void Reset()
        {
            CreatedOutsideScopeCount = 0;
            DisposedOutsideScopeCount = 0;
            CreatedInScopeCount = 0;
            DisposedInScopeCount = 0;
            AttachedToScopeCount = 0;
            DetachedFromScopeCount = 0;
        }
        public override string ToString()
        {
            return LifetimeStatisticsUtil.DebugString(this);
        }
    }
    static class LifetimeStatisticsUtil
    {
        public static string DebugString(this ILifetimeStatistics statistics)
        {
            var sb = new StringBuilder();
            sb.Append("ThreadTotalLiveCount: " + statistics.ThreadTotalLiveCount);
            sb.Append("; CreatedOutsideScopeCount: " + statistics.CreatedOutsideScopeCount);
            sb.Append("; DisposedOutsideScopeCount: " + statistics.DisposedOutsideScopeCount);
            sb.Append("; CreatedInScopeCount: " + statistics.CreatedInScopeCount);
            sb.Append("; DisposedInScopeCount: " + statistics.DisposedInScopeCount);
            sb.Append("; AttachedToScopeCount: " + statistics.AttachedToScopeCount);
            sb.Append("; DetachedFromScopeCount: " + statistics.DetachedFromScopeCount);
            return sb.ToString();
        }
    }
}

