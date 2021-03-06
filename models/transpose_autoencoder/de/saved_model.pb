��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8��
�
conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *+
shared_nameconv2d_transpose_27/kernel
�
.conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/kernel*&
_output_shapes
:  *
dtype0
�
conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_27/bias
�
,conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_nameconv2d_transpose_28/kernel
�
.conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/kernel*&
_output_shapes
:@ *
dtype0
�
conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_28/bias
�
,conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/bias*
_output_shapes
:@*
dtype0
�
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_39/kernel
}
$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_39/bias
m
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
*
	0

1
2
3
4
5
*
	0

1
2
3
4
5
�
metrics
regularization_losses

layers
layer_metrics
trainable_variables
non_trainable_variables
layer_regularization_losses
	variables
 
fd
VARIABLE_VALUEconv2d_transpose_27/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_27/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
�
 metrics
regularization_losses

!layers
"layer_metrics
trainable_variables
#non_trainable_variables
$layer_regularization_losses
	variables
fd
VARIABLE_VALUEconv2d_transpose_28/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_28/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
%metrics
regularization_losses

&layers
'layer_metrics
trainable_variables
(non_trainable_variables
)layer_regularization_losses
	variables
\Z
VARIABLE_VALUEconv2d_39/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_39/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
*metrics
regularization_losses

+layers
,layer_metrics
trainable_variables
-non_trainable_variables
.layer_regularization_losses
	variables
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_9Placeholder*/
_output_shapes
:��������� *
dtype0*$
shape:��������� 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9conv2d_transpose_27/kernelconv2d_transpose_27/biasconv2d_transpose_28/kernelconv2d_transpose_28/biasconv2d_39/kernelconv2d_39/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1237247
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.conv2d_transpose_27/kernel/Read/ReadVariableOp,conv2d_transpose_27/bias/Read/ReadVariableOp.conv2d_transpose_28/kernel/Read/ReadVariableOp,conv2d_transpose_28/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1237444
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_transpose_27/kernelconv2d_transpose_27/biasconv2d_transpose_28/kernelconv2d_transpose_28/biasconv2d_39/kernelconv2d_39/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1237472ߪ
�
�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237213

inputs
conv2d_transpose_27_1237197
conv2d_transpose_27_1237199
conv2d_transpose_28_1237202
conv2d_transpose_28_1237204
conv2d_39_1237207
conv2d_39_1237209
identity��!conv2d_39/StatefulPartitionedCall�+conv2d_transpose_27/StatefulPartitionedCall�+conv2d_transpose_28/StatefulPartitionedCall�
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_27_1237197conv2d_transpose_27_1237199*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_12370392-
+conv2d_transpose_27/StatefulPartitionedCall�
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_1237202conv2d_transpose_28_1237204*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_12370842-
+conv2d_transpose_28/StatefulPartitionedCall�
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_39_1237207conv2d_39_1237209*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_12371192#
!conv2d_39/StatefulPartitionedCall�
IdentityIdentity*conv2d_39/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237136
input_9
conv2d_transpose_27_1237098
conv2d_transpose_27_1237100
conv2d_transpose_28_1237103
conv2d_transpose_28_1237105
conv2d_39_1237130
conv2d_39_1237132
identity��!conv2d_39/StatefulPartitionedCall�+conv2d_transpose_27/StatefulPartitionedCall�+conv2d_transpose_28/StatefulPartitionedCall�
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_transpose_27_1237098conv2d_transpose_27_1237100*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_12370392-
+conv2d_transpose_27/StatefulPartitionedCall�
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_1237103conv2d_transpose_28_1237105*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_12370842-
+conv2d_transpose_28/StatefulPartitionedCall�
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_39_1237130conv2d_39_1237132*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_12371192#
!conv2d_39/StatefulPartitionedCall�
IdentityIdentity*conv2d_39/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_9
�
�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237177

inputs
conv2d_transpose_27_1237161
conv2d_transpose_27_1237163
conv2d_transpose_28_1237166
conv2d_transpose_28_1237168
conv2d_39_1237171
conv2d_39_1237173
identity��!conv2d_39/StatefulPartitionedCall�+conv2d_transpose_27/StatefulPartitionedCall�+conv2d_transpose_28/StatefulPartitionedCall�
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_27_1237161conv2d_transpose_27_1237163*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_12370392-
+conv2d_transpose_27/StatefulPartitionedCall�
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_1237166conv2d_transpose_28_1237168*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_12370842-
+conv2d_transpose_28/StatefulPartitionedCall�
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_39_1237171conv2d_39_1237173*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_12371192#
!conv2d_39/StatefulPartitionedCall�
IdentityIdentity*conv2d_39/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1237119

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1237394

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237155
input_9
conv2d_transpose_27_1237139
conv2d_transpose_27_1237141
conv2d_transpose_28_1237144
conv2d_transpose_28_1237146
conv2d_39_1237149
conv2d_39_1237151
identity��!conv2d_39/StatefulPartitionedCall�+conv2d_transpose_27/StatefulPartitionedCall�+conv2d_transpose_28/StatefulPartitionedCall�
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_transpose_27_1237139conv2d_transpose_27_1237141*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_12370392-
+conv2d_transpose_27/StatefulPartitionedCall�
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_1237144conv2d_transpose_28_1237146*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_12370842-
+conv2d_transpose_28/StatefulPartitionedCall�
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_39_1237149conv2d_39_1237151*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_12371192#
!conv2d_39/StatefulPartitionedCall�
IdentityIdentity*conv2d_39/StatefulPartitionedCall:output:0"^conv2d_39/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_9
�_
�
"__inference__wrapped_model_1237004
input_9N
Jsequential_23_conv2d_transpose_27_conv2d_transpose_readvariableop_resourceE
Asequential_23_conv2d_transpose_27_biasadd_readvariableop_resourceN
Jsequential_23_conv2d_transpose_28_conv2d_transpose_readvariableop_resourceE
Asequential_23_conv2d_transpose_28_biasadd_readvariableop_resource:
6sequential_23_conv2d_39_conv2d_readvariableop_resource;
7sequential_23_conv2d_39_biasadd_readvariableop_resource
identity��.sequential_23/conv2d_39/BiasAdd/ReadVariableOp�-sequential_23/conv2d_39/Conv2D/ReadVariableOp�8sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOp�Asequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOp�8sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOp�Asequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOp�
'sequential_23/conv2d_transpose_27/ShapeShapeinput_9*
T0*
_output_shapes
:2)
'sequential_23/conv2d_transpose_27/Shape�
5sequential_23/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_23/conv2d_transpose_27/strided_slice/stack�
7sequential_23/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_23/conv2d_transpose_27/strided_slice/stack_1�
7sequential_23/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_23/conv2d_transpose_27/strided_slice/stack_2�
/sequential_23/conv2d_transpose_27/strided_sliceStridedSlice0sequential_23/conv2d_transpose_27/Shape:output:0>sequential_23/conv2d_transpose_27/strided_slice/stack:output:0@sequential_23/conv2d_transpose_27/strided_slice/stack_1:output:0@sequential_23/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_23/conv2d_transpose_27/strided_slice�
)sequential_23/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_23/conv2d_transpose_27/stack/1�
)sequential_23/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_23/conv2d_transpose_27/stack/2�
)sequential_23/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_23/conv2d_transpose_27/stack/3�
'sequential_23/conv2d_transpose_27/stackPack8sequential_23/conv2d_transpose_27/strided_slice:output:02sequential_23/conv2d_transpose_27/stack/1:output:02sequential_23/conv2d_transpose_27/stack/2:output:02sequential_23/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_23/conv2d_transpose_27/stack�
7sequential_23/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_23/conv2d_transpose_27/strided_slice_1/stack�
9sequential_23/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_23/conv2d_transpose_27/strided_slice_1/stack_1�
9sequential_23/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_23/conv2d_transpose_27/strided_slice_1/stack_2�
1sequential_23/conv2d_transpose_27/strided_slice_1StridedSlice0sequential_23/conv2d_transpose_27/stack:output:0@sequential_23/conv2d_transpose_27/strided_slice_1/stack:output:0Bsequential_23/conv2d_transpose_27/strided_slice_1/stack_1:output:0Bsequential_23/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_23/conv2d_transpose_27/strided_slice_1�
Asequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_23_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02C
Asequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOp�
2sequential_23/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput0sequential_23/conv2d_transpose_27/stack:output:0Isequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0input_9*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
24
2sequential_23/conv2d_transpose_27/conv2d_transpose�
8sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOpAsequential_23_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOp�
)sequential_23/conv2d_transpose_27/BiasAddBiasAdd;sequential_23/conv2d_transpose_27/conv2d_transpose:output:0@sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2+
)sequential_23/conv2d_transpose_27/BiasAdd�
&sequential_23/conv2d_transpose_27/ReluRelu2sequential_23/conv2d_transpose_27/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2(
&sequential_23/conv2d_transpose_27/Relu�
'sequential_23/conv2d_transpose_28/ShapeShape4sequential_23/conv2d_transpose_27/Relu:activations:0*
T0*
_output_shapes
:2)
'sequential_23/conv2d_transpose_28/Shape�
5sequential_23/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_23/conv2d_transpose_28/strided_slice/stack�
7sequential_23/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_23/conv2d_transpose_28/strided_slice/stack_1�
7sequential_23/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_23/conv2d_transpose_28/strided_slice/stack_2�
/sequential_23/conv2d_transpose_28/strided_sliceStridedSlice0sequential_23/conv2d_transpose_28/Shape:output:0>sequential_23/conv2d_transpose_28/strided_slice/stack:output:0@sequential_23/conv2d_transpose_28/strided_slice/stack_1:output:0@sequential_23/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_23/conv2d_transpose_28/strided_slice�
)sequential_23/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_23/conv2d_transpose_28/stack/1�
)sequential_23/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_23/conv2d_transpose_28/stack/2�
)sequential_23/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_23/conv2d_transpose_28/stack/3�
'sequential_23/conv2d_transpose_28/stackPack8sequential_23/conv2d_transpose_28/strided_slice:output:02sequential_23/conv2d_transpose_28/stack/1:output:02sequential_23/conv2d_transpose_28/stack/2:output:02sequential_23/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_23/conv2d_transpose_28/stack�
7sequential_23/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_23/conv2d_transpose_28/strided_slice_1/stack�
9sequential_23/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_23/conv2d_transpose_28/strided_slice_1/stack_1�
9sequential_23/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_23/conv2d_transpose_28/strided_slice_1/stack_2�
1sequential_23/conv2d_transpose_28/strided_slice_1StridedSlice0sequential_23/conv2d_transpose_28/stack:output:0@sequential_23/conv2d_transpose_28/strided_slice_1/stack:output:0Bsequential_23/conv2d_transpose_28/strided_slice_1/stack_1:output:0Bsequential_23/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_23/conv2d_transpose_28/strided_slice_1�
Asequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_23_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02C
Asequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOp�
2sequential_23/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput0sequential_23/conv2d_transpose_28/stack:output:0Isequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:04sequential_23/conv2d_transpose_27/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
24
2sequential_23/conv2d_transpose_28/conv2d_transpose�
8sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOpAsequential_23_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOp�
)sequential_23/conv2d_transpose_28/BiasAddBiasAdd;sequential_23/conv2d_transpose_28/conv2d_transpose:output:0@sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2+
)sequential_23/conv2d_transpose_28/BiasAdd�
&sequential_23/conv2d_transpose_28/ReluRelu2sequential_23/conv2d_transpose_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2(
&sequential_23/conv2d_transpose_28/Relu�
-sequential_23/conv2d_39/Conv2D/ReadVariableOpReadVariableOp6sequential_23_conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02/
-sequential_23/conv2d_39/Conv2D/ReadVariableOp�
sequential_23/conv2d_39/Conv2DConv2D4sequential_23/conv2d_transpose_28/Relu:activations:05sequential_23/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
2 
sequential_23/conv2d_39/Conv2D�
.sequential_23/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp7sequential_23_conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_23/conv2d_39/BiasAdd/ReadVariableOp�
sequential_23/conv2d_39/BiasAddBiasAdd'sequential_23/conv2d_39/Conv2D:output:06sequential_23/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
sequential_23/conv2d_39/BiasAdd�
sequential_23/conv2d_39/SigmoidSigmoid(sequential_23/conv2d_39/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2!
sequential_23/conv2d_39/Sigmoid�
IdentityIdentity#sequential_23/conv2d_39/Sigmoid:y:0/^sequential_23/conv2d_39/BiasAdd/ReadVariableOp.^sequential_23/conv2d_39/Conv2D/ReadVariableOp9^sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOpB^sequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOp9^sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOpB^sequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2`
.sequential_23/conv2d_39/BiasAdd/ReadVariableOp.sequential_23/conv2d_39/BiasAdd/ReadVariableOp2^
-sequential_23/conv2d_39/Conv2D/ReadVariableOp-sequential_23/conv2d_39/Conv2D/ReadVariableOp2t
8sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOp8sequential_23/conv2d_transpose_27/BiasAdd/ReadVariableOp2�
Asequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOpAsequential_23/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2t
8sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOp8sequential_23/conv2d_transpose_28/BiasAdd/ReadVariableOp2�
Asequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOpAsequential_23/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_9
�M
�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237298

inputs@
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_27_biasadd_readvariableop_resource@
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_28_biasadd_readvariableop_resource,
(conv2d_39_conv2d_readvariableop_resource-
)conv2d_39_biasadd_readvariableop_resource
identity�� conv2d_39/BiasAdd/ReadVariableOp�conv2d_39/Conv2D/ReadVariableOp�*conv2d_transpose_27/BiasAdd/ReadVariableOp�3conv2d_transpose_27/conv2d_transpose/ReadVariableOp�*conv2d_transpose_28/BiasAdd/ReadVariableOp�3conv2d_transpose_28/conv2d_transpose/ReadVariableOpl
conv2d_transpose_27/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_27/Shape�
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_27/strided_slice/stack�
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_1�
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_2�
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_27/strided_slice|
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/stack/1|
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/stack/2|
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_27/stack/3�
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_27/stack�
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_27/strided_slice_1/stack�
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_1�
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_2�
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_1�
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype025
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2&
$conv2d_transpose_27/conv2d_transpose�
*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_27/BiasAdd/ReadVariableOp�
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_27/BiasAdd�
conv2d_transpose_27/ReluRelu$conv2d_transpose_27/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_27/Relu�
conv2d_transpose_28/ShapeShape&conv2d_transpose_27/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_28/Shape�
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_28/strided_slice/stack�
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_1�
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_2�
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_28/strided_slice|
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_28/stack/1|
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_28/stack/2|
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_28/stack/3�
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_28/stack�
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_28/strided_slice_1/stack�
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_1�
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_2�
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_1�
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype025
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_27/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2&
$conv2d_transpose_28/conv2d_transpose�
*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_28/BiasAdd/ReadVariableOp�
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
conv2d_transpose_28/BiasAdd�
conv2d_transpose_28/ReluRelu$conv2d_transpose_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
conv2d_transpose_28/Relu�
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_39/Conv2D/ReadVariableOp�
conv2d_39/Conv2DConv2D&conv2d_transpose_28/Relu:activations:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
2
conv2d_39/Conv2D�
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp�
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_39/BiasAdd�
conv2d_39/SigmoidSigmoidconv2d_39/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_39/Sigmoid�
IdentityIdentityconv2d_39/Sigmoid:y:0!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�$
�
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_1237084

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_23_layer_call_fn_1237366

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_12371772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_23_layer_call_fn_1237228
input_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_12372132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_9
�$
�
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_1237039

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�M
�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237349

inputs@
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_27_biasadd_readvariableop_resource@
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_28_biasadd_readvariableop_resource,
(conv2d_39_conv2d_readvariableop_resource-
)conv2d_39_biasadd_readvariableop_resource
identity�� conv2d_39/BiasAdd/ReadVariableOp�conv2d_39/Conv2D/ReadVariableOp�*conv2d_transpose_27/BiasAdd/ReadVariableOp�3conv2d_transpose_27/conv2d_transpose/ReadVariableOp�*conv2d_transpose_28/BiasAdd/ReadVariableOp�3conv2d_transpose_28/conv2d_transpose/ReadVariableOpl
conv2d_transpose_27/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_27/Shape�
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_27/strided_slice/stack�
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_1�
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_27/strided_slice/stack_2�
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_27/strided_slice|
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/stack/1|
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_27/stack/2|
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_27/stack/3�
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_27/stack�
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_27/strided_slice_1/stack�
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_1�
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_27/strided_slice_1/stack_2�
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_27/strided_slice_1�
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype025
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2&
$conv2d_transpose_27/conv2d_transpose�
*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_27/BiasAdd/ReadVariableOp�
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_27/BiasAdd�
conv2d_transpose_27/ReluRelu$conv2d_transpose_27/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_27/Relu�
conv2d_transpose_28/ShapeShape&conv2d_transpose_27/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_28/Shape�
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_28/strided_slice/stack�
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_1�
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_28/strided_slice/stack_2�
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_28/strided_slice|
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_28/stack/1|
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_28/stack/2|
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_28/stack/3�
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_28/stack�
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_28/strided_slice_1/stack�
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_1�
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_28/strided_slice_1/stack_2�
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_28/strided_slice_1�
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype025
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp�
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_27/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
2&
$conv2d_transpose_28/conv2d_transpose�
*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_28/BiasAdd/ReadVariableOp�
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @2
conv2d_transpose_28/BiasAdd�
conv2d_transpose_28/ReluRelu$conv2d_transpose_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @2
conv2d_transpose_28/Relu�
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_39/Conv2D/ReadVariableOp�
conv2d_39/Conv2DConv2D&conv2d_transpose_28/Relu:activations:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
2
conv2d_39/Conv2D�
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp�
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_39/BiasAdd�
conv2d_39/SigmoidSigmoidconv2d_39/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_39/Sigmoid�
IdentityIdentityconv2d_39/Sigmoid:y:0!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_23_layer_call_fn_1237192
input_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_12371772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_9
�
�
5__inference_conv2d_transpose_27_layer_call_fn_1237049

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_12370392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
#__inference__traced_restore_1237472
file_prefix/
+assignvariableop_conv2d_transpose_27_kernel/
+assignvariableop_1_conv2d_transpose_27_bias1
-assignvariableop_2_conv2d_transpose_28_kernel/
+assignvariableop_3_conv2d_transpose_28_bias'
#assignvariableop_4_conv2d_39_kernel%
!assignvariableop_5_conv2d_39_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp+assignvariableop_conv2d_transpose_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_conv2d_transpose_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv2d_transpose_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv2d_transpose_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_39_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_39_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_conv2d_39_layer_call_fn_1237403

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_39_layer_call_and_return_conditional_losses_12371192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
5__inference_conv2d_transpose_28_layer_call_fn_1237094

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_12370842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
 __inference__traced_save_1237444
file_prefix9
5savev2_conv2d_transpose_27_kernel_read_readvariableop7
3savev2_conv2d_transpose_27_bias_read_readvariableop9
5savev2_conv2d_transpose_28_kernel_read_readvariableop7
3savev2_conv2d_transpose_28_bias_read_readvariableop/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_conv2d_transpose_27_kernel_read_readvariableop3savev2_conv2d_transpose_27_bias_read_readvariableop5savev2_conv2d_transpose_28_kernel_read_readvariableop3savev2_conv2d_transpose_28_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*_
_input_shapesN
L: :  : :@ :@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:@ : 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::

_output_shapes
: 
�
�
/__inference_sequential_23_layer_call_fn_1237383

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_12372132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1237247
input_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_12370042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_9"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_98
serving_default_input_9:0��������� E
	conv2d_398
StatefulPartitionedCall:0���������  tensorflow/serving/predict:��
�+
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
/_default_save_signature
0__call__
*1&call_and_return_all_conditional_losses"�)
_tf_keras_sequential�({"class_name": "Sequential", "name": "sequential_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�


	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
�


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
�	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
 "
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
�
metrics
regularization_losses

layers
layer_metrics
trainable_variables
non_trainable_variables
layer_regularization_losses
	variables
0__call__
/_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
4:2  2conv2d_transpose_27/kernel
&:$ 2conv2d_transpose_27/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
�
 metrics
regularization_losses

!layers
"layer_metrics
trainable_variables
#non_trainable_variables
$layer_regularization_losses
	variables
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
4:2@ 2conv2d_transpose_28/kernel
&:$@2conv2d_transpose_28/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
%metrics
regularization_losses

&layers
'layer_metrics
trainable_variables
(non_trainable_variables
)layer_regularization_losses
	variables
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_39/kernel
:2conv2d_39/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
*metrics
regularization_losses

+layers
,layer_metrics
trainable_variables
-non_trainable_variables
.layer_regularization_losses
	variables
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
"__inference__wrapped_model_1237004�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_9��������� 
�2�
/__inference_sequential_23_layer_call_fn_1237228
/__inference_sequential_23_layer_call_fn_1237366
/__inference_sequential_23_layer_call_fn_1237192
/__inference_sequential_23_layer_call_fn_1237383�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237136
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237349
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237298
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237155�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
5__inference_conv2d_transpose_27_layer_call_fn_1237049�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_1237039�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
5__inference_conv2d_transpose_28_layer_call_fn_1237094�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_1237084�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
+__inference_conv2d_39_layer_call_fn_1237403�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1237394�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1237247input_9"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_1237004�	
8�5
.�+
)�&
input_9��������� 
� "=�:
8
	conv2d_39+�(
	conv2d_39���������  �
F__inference_conv2d_39_layer_call_and_return_conditional_losses_1237394�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������
� �
+__inference_conv2d_39_layer_call_fn_1237403�I�F
?�<
:�7
inputs+���������������������������@
� "2�/+����������������������������
P__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_1237039�	
I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
5__inference_conv2d_transpose_27_layer_call_fn_1237049�	
I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
P__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_1237084�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
5__inference_conv2d_transpose_28_layer_call_fn_1237094�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237136�	
@�=
6�3
)�&
input_9��������� 
p

 
� "?�<
5�2
0+���������������������������
� �
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237155�	
@�=
6�3
)�&
input_9��������� 
p 

 
� "?�<
5�2
0+���������������������������
� �
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237298x	
?�<
5�2
(�%
inputs��������� 
p

 
� "-�*
#� 
0���������  
� �
J__inference_sequential_23_layer_call_and_return_conditional_losses_1237349x	
?�<
5�2
(�%
inputs��������� 
p 

 
� "-�*
#� 
0���������  
� �
/__inference_sequential_23_layer_call_fn_1237192~	
@�=
6�3
)�&
input_9��������� 
p

 
� "2�/+����������������������������
/__inference_sequential_23_layer_call_fn_1237228~	
@�=
6�3
)�&
input_9��������� 
p 

 
� "2�/+����������������������������
/__inference_sequential_23_layer_call_fn_1237366}	
?�<
5�2
(�%
inputs��������� 
p

 
� "2�/+����������������������������
/__inference_sequential_23_layer_call_fn_1237383}	
?�<
5�2
(�%
inputs��������� 
p 

 
� "2�/+����������������������������
%__inference_signature_wrapper_1237247�	
C�@
� 
9�6
4
input_9)�&
input_9��������� "=�:
8
	conv2d_39+�(
	conv2d_39���������  