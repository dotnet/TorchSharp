# Memory Leak Troubleshooting

If suspect you are leaking memory this is your guide. First be sure to be familiar with the [Memory Management Techniques](memory.md).

## Verifying you have a leak

The `DisposeScopeManager.Statistics` property defines thread level statistics of objects captured
in TorchSharp as objects are created and moved between DisposeScopes. Normally deal directly
with only this property.

To see where code may be leaking objects, it is easiest to modify the training loop.
Use a DisposeScope, reset the global statistics to have a known starting point, then take
some action and look at the statistics to see what's still around.

```csharp
	//Training Loop, 10 epochs
	for (int i = 0; i < 10; i++) {
		//Clear the statistics
		DisposeScopeManager.Statistics.Reset();
		//Take action. In this case it is inside a DisposeScope, so
		//when this code block is done, there should be no new live objects.
		using (NewDisposeScope()) {
			var eval = model.call(x);
			// ... other model execution code
			optimizer.step();
		}
		//Examine what happened
		Console.WriteLine(DisposeScopeManager.Statistics);
	}
```

If on every iteration the number of live objects is increasing, there is a leak. In the following
example note that the number of live objects increases by 200 every iteration. It can also be
seen these objects were created on a DisposeScope, but were eventually detached. In this specific
case, look for where the code is detaching the tensors, and then determine how
to correctly manage the lifetime of these objects.
```csharp
ThreadTotalLiveCount: 548; CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; CreatedInScopeCount: 200; DisposedInScopeCount: 2; AttachedToScopeCount: 0; DetachedFromScopeCount: 200"
ThreadTotalLiveCount: 748; CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; CreatedInScopeCount: 200; DisposedInScopeCount: 2; AttachedToScopeCount: 0; DetachedFromScopeCount: 200"
ThreadTotalLiveCount: 948; CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; CreatedInScopeCount: 200; DisposedInScopeCount: 2; AttachedToScopeCount: 0; DetachedFromScopeCount: 200"
```

It is not necessary to leave this code in place for production implementations after fixing a leak. It may be removed so the code looks more Pythonic if needed.

## Identifying the leak
This is where the leg work is. Look at each line of code where Tensor or PackedSequence objects are created.
Ensure they are eventually disposed either manually or by a DisposeScope. One can also print the statistics to
the debugger while stepping code for an interactive approach.

Be aware that TorchSharp also creates tensors for itself and uses them in various
ways. Just because one finds a tensor that is created by TorchSharp isn't being disposed, it likely isn't caused
by TorchSharp. A good example is the Adam optimizer. It creates tensors internally to manage it's parameters,
and detaches them from any DisposeScope that is in use. If it didn't, it would fail doing gradients and back
propagation as it's tensors would have been disposed. These are eventually cleaned up when the optimizer is
properly disposed after training. Faliure of the client code to dispose is the most likely cause of memory leaks.

## Working with RNNs
One may want to drill down to `DisposeScopeManager.Statistics.TensorStatistics` or
`DisposeScopeManager.Statistics.PackedSequenceStatistics`, these track Tensors and
PackedSequence usages independently.

Additionally, a PackedSequence uses some tensors internally. These tensors show up in the creation statistics,
and are immediately detached from any scope if there is one in context and will
increment the DetachedFromScopeCount property. When a PackedSequence is disposed, it will also Dispose
it's tensors. The differences in counts can be seen in the following, which represents output within an IDE
debug window where all three levels of statistics were observed at the same execution time. Note the first two
sum to the totals on the last line.

```
  DisposeScopeManager.Statistics.TensorStatistics.ToString()
  "ThreadTotalLiveCount: 4; CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; CreatedInScopeCount: 6; DisposedInScopeCount: 2; AttachedToScopeCount: 0; DetachedFromScopeCount: 4"

  DisposeScopeManager.Statistics.PackedSequenceStatistics.ToString()
  "ThreadTotalLiveCount: 1; CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; CreatedInScopeCount: 1; DisposedInScopeCount: 0; AttachedToScopeCount: 0; DetachedFromScopeCount: 0"

  DisposeScopeManager.Statistics.ToString()
  "ThreadTotalLiveCount: 5; CreatedOutsideScopeCount: 0; DisposedOutsideScopeCount: 0; CreatedInScopeCount: 7; DisposedInScopeCount: 2; AttachedToScopeCount: 0; DetachedFromScopeCount: 4"

```


