#nullable enable
using System.Text;

namespace TorchSharp
{

    /// <summary>
    /// Keeps track of the combined Tensor and PackedSequence statistics for the current thread. Can be queried to figure out performance/memory issues.
    /// </summary>
    public class ThreadDisposeScopeStatistics
    {
        /// <summary>
        /// The total number of Tensors and PackedSequence instances that were created on this thread, but weren't
        /// captured by a DisposeScope.
        /// </summary>
        public long CreatedOutsideScopeCount => TensorStatistics.CreatedOutsideScopeCount +
                                                PackedSequenceStatistics.CreatedOutsideScopeCount;

        /// <summary>
        /// The number of Tensors and PackedSequence instances that were Disposed on this thread, but weren't
        /// captured by a DisposeScope.
        /// </summary>
        public long DisposedOutsideScopeCount => TensorStatistics.DisposedOutsideScopeCount +
                                                 PackedSequenceStatistics.DisposedOutsideScopeCount;
        /// <summary>
        /// The number of Tensors and PackedSequence instances that were created on this thread and were captured
        /// by a DisposeScope.
        /// </summary>
        public long CreatedInScopeCount => TensorStatistics.CreatedInScopeCount +
                                           PackedSequenceStatistics.CreatedInScopeCount;

        /// <summary>
        /// The number of Tensors and PackedSequence instances that were disposed on this thread and were disposed
        /// while in a DisposeScope.
        /// </summary>
        public long DisposedInScopeCount => TensorStatistics.DisposedInScopeCount +
                                            PackedSequenceStatistics.DisposedInScopeCount;

        /// <summary>
        /// The number of Tensors and PackedSequence instances that were created on this thread outside a DisposeScope,
        /// and then eventually were attached to one.
        /// </summary>
        public long AttachedToScopeCount=> TensorStatistics.AttachedToScopeCount +
                                           PackedSequenceStatistics.AttachedToScopeCount;

        /// <summary>
        /// Number of Tensors and PackedSequence instances that were once included in a DisposeScope, but were
        /// subsequently detached.
        /// </summary>
        public long DetachedFromScopeCount=> TensorStatistics.DetachedFromScopeCount +
                                             PackedSequenceStatistics.DetachedFromScopeCount;

        /// <summary>
        /// Exact number of Tensors and PackedSequence instances that are still live. Is the difference of all
        /// created objects minus all disposed objects.
        /// </summary>
        public long ThreadTotalLiveCount => TensorStatistics.ThreadTotalLiveCount +
                                            PackedSequenceStatistics.ThreadTotalLiveCount;

        /// <summary>
        /// Resets the counts for the current thread. See ThreadTotalLiveCount etc. Mainly used in tests and memory
        /// leak debugging to make sure we get a clean slate on the thread.
        /// </summary>
        public void Reset()
        {
            TensorStatistics.Reset();
            PackedSequenceStatistics.Reset();
        }
        /// <summary>
        /// A debug printout of all the properties and their values, suitable for a log or console output
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("ThreadTotalLiveCount: " + ThreadTotalLiveCount);
            sb.Append("; CreatedOutsideScopeCount: " + CreatedOutsideScopeCount);
            sb.Append("; DisposedOutsideScopeCount: " + DisposedOutsideScopeCount);
            sb.Append("; CreatedInScopeCount: " + CreatedInScopeCount);
            sb.Append("; DisposedInScopeCount: " + DisposedInScopeCount);
            sb.Append("; AttachedToScopeCount: " + AttachedToScopeCount);
            sb.Append("; DetachedFromScopeCount: " + DetachedFromScopeCount);
            return sb.ToString();
        }
        /// <summary>
        /// Keeps track of the Tensor statistics for the current thread. Can be queried to figure out performance/memory issues.
        /// </summary>
        public LifetimeStatistics TensorStatistics { get; } = new LifetimeStatistics();
        /// <summary>
        /// Keeps track of the PackedSequence statistics for the current thread. Can be queried to figure out performance/memory issues.
        /// </summary>
        public LifetimeStatistics PackedSequenceStatistics { get; } = new LifetimeStatistics();
    }

    public class LifetimeStatistics
    {
        /// <summary>
        /// The number of disposables that were created on this thread, but weren't captured by a DisposeScope.
        /// </summary>
        public long CreatedOutsideScopeCount { get; internal set; }
        /// <summary>
        /// The number of disposables that were Disposed on this thread, but weren't captured by a DisposeScope.
        /// </summary>
        public long DisposedOutsideScopeCount { get; internal set; }
        /// <summary>
        /// The number of disposables that were created on this thread and were captured by a DisposeScope.
        /// </summary>
        public long CreatedInScopeCount { get; internal set; }
        /// <summary>
        /// The number of disposables that were disposed on this thread and were disposed while in a DisposeScope.
        /// </summary>
        public long DisposedInScopeCount { get; internal set; }
        /// <summary>
        /// The number of disposables that were created on this thread outside a DisposeScope, and then
        /// eventually were attached to one.
        /// </summary>
        public long AttachedToScopeCount { get; internal set; }
        /// <summary>
        /// Number of disposables that were once included in a DisposeScope, but were subsequently detached.
        /// </summary>
        public long DetachedFromScopeCount { get; internal set; }
        /// <summary>
        /// Exact number of objects that are still live. Is the difference of all created objects
        /// minus all disposed objects.
        /// </summary>
        public long ThreadTotalLiveCount => (CreatedInScopeCount - DisposedInScopeCount) +
                                            (CreatedOutsideScopeCount - DisposedOutsideScopeCount);

        /// <summary>
        /// Resets the counts for the current thread. See ThreadTotalLiveCount etc. Mainly used in tests to make sure
        /// we get a clean slate on the thread.
        /// </summary>
        public void Reset()
        {
            CreatedOutsideScopeCount = 0;
            DisposedOutsideScopeCount = 0;
            CreatedInScopeCount = 0;
            DisposedInScopeCount = 0;
            AttachedToScopeCount = 0;
            DetachedFromScopeCount = 0;
        }
        /// <summary>
        /// A debug printout of all the properties and their values, suitable for a log or console output
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("ThreadTotalLiveCount: " + ThreadTotalLiveCount);
            sb.Append("; CreatedOutsideScopeCount: " + CreatedOutsideScopeCount);
            sb.Append("; DisposedOutsideScopeCount: " + DisposedOutsideScopeCount);
            sb.Append("; CreatedInScopeCount: " + CreatedInScopeCount);
            sb.Append("; DisposedInScopeCount: " + DisposedInScopeCount);
            sb.Append("; AttachedToScopeCount: " + AttachedToScopeCount);
            sb.Append("; DetachedFromScopeCount: " + DetachedFromScopeCount);
            return sb.ToString();
        }
    }
}

