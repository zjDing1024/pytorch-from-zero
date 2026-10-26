# Official source study

Date: 2026-10-14, Asia/Kuala_Lumpur. Project: [PyTorch](https://github.com/pytorch/pytorch), maintained by PyTorch contributors. The study supports the present CPU/autograd stage of the AI engineering portfolio. Its Python API, ATen operators, derivative specifications and CPU kernels make the saved-forward/backward boundary inspectable. Original implementation, no upstream code copied.

Installed Python 3.12.14/PyTorch 2.14.1+cpu revision `5c4886908584029761b579af026dcfb627c84070` was verified. Actual-runtime graph.py and functional.py Git blob hashes exactly match the official pinned revision. The installed LogSoftmaxKernelImpl.h differs only by a packaging guard. Source agreement is not a reproducible-build or executed-microkernel proof.

## Public hook contract

Pinned [graph.py, lines 273–337](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/autograd/graph.py#L273-L337) documents:

- Pack is invoked when an operation saves a tensor for backward within the context, including PyTorch-defined operators.
- Its return is kept by the graph; unpack later receives that return and provides a Tensor.
- The unpacked Tensor should match original value, size, dtype and device.
- In-place modification of hook inputs is unsafe.
- Only the innermost nested hook pair applies.
- The pack return must not retain the original input Tensor reference; the documented example uses detach.

The [2.14 autograd mechanics documentation](https://docs.pytorch.org/docs/2.14/notes/autograd.html#hooks-for-saved-tensors) additionally explains that unpack can run more than once and must return the same data each time; the context is thread-local; autograd metadata is restored during unpack; and hooks alter tensor-object reuse. These are public documentation claims, not results from this experiment.

Recommended observer structure, expressed as design rather than copied implementation:

- At pack, create `payload = tensor.detach()`.
- Save a separately allocated `payload.clone()` for evidence. Record metadata of the original tensor before canonicalizing evidence bytes; original stride/layout are not the clone's canonical byte layout.
- Return payload, or a small structure containing payload and an integer record ID. Never include the original live tensor.
- At unpack, record the event if needed and return that same payload without any value-changing operation.
- Never mutate payload or its aliases between saving and use. Detach shares underlying storage. Evidence clones must not later be used as in-place work buffers.
- Keep hook context limited to the original forward being studied so manual recomputations do not contaminate pack records.

Pack events represent saved slots, not unique tensors. Two saved slots can contain byte-identical tensors, or share storage. Preserve duplicates and separate pack sequence IDs. The public callback supplies a tensor, not a stable operator/argument label; shape or value matching alone is insufficient to claim unique provenance. Stage-local context plus pinned derivative rules can support a qualified attribution.

Pinned [SavedVariable, lines 165–210](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/saved_variable.cpp#L165-L210) confirms that the normal version-counter comparison is only performed without hooks. This is a source-reading finding; the probe need not call this internal implementation or any private API.

Required controls should compare the original no-hook run with the detached-value observer run for the same frozen inputs, initial parameters, original loss call, gradients, first-step optimizer state and updated parameters. Passing these controls supports this experiment only. It does not turn hooks into a generally transparent observer for arbitrary programs.

## Cross-entropy and saved derivative inputs

Pinned [functional.py, lines 3478–3487 and 3546–3569](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py#L3478-L3569) exposes the public default mean reduction, no class weight, ignore_index -100 and label_smoothing 0.0; it calls the native cross-entropy binding. The probe should continue calling public `F.cross_entropy`, not the binding directly.

Pinned [LossNLL.cpp, lines 632–675](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp#L632-L675):

- Equal input/target shapes select probability-target handling.
- Positive smoothing selects a separate smoothing branch.
- Otherwise, class dimension is 0 for one-dimensional logits and 1 for higher-dimensional logits.
- That last branch computes log-softmax in the input scalar type and passes the result to NLL.
- NLL forward produces the visible loss and a total-weight tensor.

Pinned [derivatives.yaml, lines 2200–2202](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml#L2200-L2202) routes log-softmax's gradient through its saved result, reduction dimension and input scalar type. It does not reconstruct the result from saved logits.

Pinned [derivatives.yaml, lines 2107–2110](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml#L2107-L2110) passes NLL's self, target, optional weight, reduction, ignore index and total_weight to backward. Target is non-differentiable. A captured log-softmax output may therefore occur more than once among saved slots; treat this as an expected source-level possibility and verify the actual records rather than hard-coding their count.

Pinned [LossNLL.cpp, lines 345–433](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp#L345-L433) initializes the input gradient to zeros. For mean-reduced 2-D class-index loss, it puts the negative upstream scalar divided by total_weight, multiplied by a class weight if present, at each non-ignored target position. It uses input shape/dtype information but does not read input element values here. Without class weights, [lines 288–305](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp#L288-L305) makes total_weight equal to the number of non-ignored batch entries.

Related original-network evidence:

- [ReLU derivative, lines 2131–2133](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml#L2131-L2133): gates backward using the saved result.
- [addmm derivatives, lines 256–260](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml#L256-L260): matrix gradient rules require the counterpart matrix and size/stride/layout metadata. Which tensors are actually saved depends on required gradients; do not assert a fixed saved inventory from this file alone.

## CPU log-softmax arithmetic paths

Pinned [SoftMax.cpp, lines 330–354](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/SoftMax.cpp#L330-L354) makes input contiguous and selects a last-dimension kernel when the reduction is on the last dimension.

Pinned [SoftMax.cpp, lines 386–407](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/SoftMax.cpp#L386-L407) makes upstream gradient and saved output contiguous, then likewise selects last-dimension or general backward. For ordinary 2-D Wine logits [N, 3], dimension 1 is the last dimension, so this source condition points to last-dimension backward.

Pinned [SoftMaxKernel.cpp, lines 158–187](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L158-L187) uses a vector reduction to sum upstream gradient and a vector map combining upstream gradient with exponentiated saved output. [Lines 1013–1022](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L1013-L1022) instantiate the log-softmax specialization. [Lines 1059–1073](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L1059-L1073) register CPU dispatch implementations including AVX512 support.

The alternate general-dimension float/double path at [lines 417–497](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L417-L497) uses chunked summation, vector exp where possible, and scalar std::exp in its scalar tail. This is a different implementation path; its existence is not evidence that this Wine case took it.

Pinned [LogSoftmaxKernelImpl.h, lines 51–95](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h#L51-L95) computes a maximum, a sum of exp(input minus max), its logarithm, then subtracts max and log-sum in that order. The source explicitly preserves subtraction order to avoid numerical problems.

Implications:

- A recorded forward log-softmax tensor is evidence of the value actually retained by autograd; an independently reconstructed tensor is only a comparison.
- Computing exp of that recorded tensor with a separate public operator still does not expose the exact scalar/vector lane operation inside the backward kernel.
- An algebraically simplified (softmax minus one-hot) divided by batch expression can differ in rounding and operation order from NLL followed by log-softmax backward.
- Matching or mismatching by a few ULPs cannot by itself identify a specific SIMD lane, exp implementation, fused operation, reduction order or compiler transformation.
- Do not invoke underscored backward operators to claim public-API reconstruction. A source-level formula written with public operations should be named a reconstruction/candidate, not an internal-kernel trace.

## License and reuse boundary

The pinned [root LICENSE, lines 59–84](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE#L59-L84) permits use and redistribution subject to retaining notices/conditions, binary-distribution notice requirements, non-endorsement and a warranty/liability disclaimer. It is BSD-3-Clause-style; it is not a declaration that every component in a wheel has that one license.

Installed wheel METADATA lists the expression `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` and many third-party license files. Reading and citing upstream source does not make an original probe an upstream implementation. If copying or redistributing upstream code or a PyTorch binary, retain the applicable notices and inspect included third-party licenses. This package saves source metadata and original explanatory notes, not a copy of the upstream implementation.

## Explicit limits

- No numerical result, equality count, ULP distance, saved-slot inventory or intervention outcome was produced by this source review.
- No private APIs, numerical probes, builds or remote mutations were used.
- The observation API is public; internal C++ and derivative files were read as evidence only.
- Saved tensors do not include every temporary used by forward or backward. Absence from pack records is not proof that a value was never computed.
- Public hooks do not label all consumers or expose arithmetic microsteps.
- Source inspection narrows plausible explanations; only the separate controlled experiment can report this fixture's actual values and effects.

