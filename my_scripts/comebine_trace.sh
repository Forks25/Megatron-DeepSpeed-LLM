#!/bin/bash -l
cd /lus/flare/projects/Aurora_deployment/eku/scaling_MDS

merge_trace --inputs \
                Sams_Megatron-DeepSpeed/trace/torch-trace-* \
            --output \
                Sams_Megatron-DeepSpeed/trace/n1_combined.json \
# merge_trace --inputs \
#                 Sams_Megatron-DeepSpeed/trace/torch-trace-0-of-48-step4.json \
#                 Sams_Megatron-DeepSpeed/trace/torch-trace-12-of-48-step4.json \
#                 Sams_Megatron-DeepSpeed/trace/torch-trace-24-of-48-step4.json \
#                 Sams_Megatron-DeepSpeed/trace/torch-trace-36-of-48-step4.json \
#             --output \
#                 Sams_Megatron-DeepSpeed/trace/n4_combined.json \