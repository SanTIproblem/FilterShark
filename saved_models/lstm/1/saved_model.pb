ü« 
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ø
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0

rnn_1/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*'
shared_namernn_1/lstm_cell/kernel

*rnn_1/lstm_cell/kernel/Read/ReadVariableOpReadVariableOprnn_1/lstm_cell/kernel* 
_output_shapes
:
È*
dtype0

 rnn_1/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*1
shared_name" rnn_1/lstm_cell/recurrent_kernel

4rnn_1/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp rnn_1/lstm_cell/recurrent_kernel*
_output_shapes
:	@*
dtype0

rnn_1/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namernn_1/lstm_cell/bias
z
(rnn_1/lstm_cell/bias/Read/ReadVariableOpReadVariableOprnn_1/lstm_cell/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
Ä
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÿ
valueõBò Bë
Ì
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
R

	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 
#
0
1
2
3
4
#
0
1
2
3
4
 
­
	variables
trainable_variables
metrics
non_trainable_variables
layer_regularization_losses

 layers
regularization_losses
!layer_metrics
 
 
 
 
­

	variables
trainable_variables
"metrics
#non_trainable_variables
$layer_regularization_losses

%layers
regularization_losses
&layer_metrics
~

kernel
recurrent_kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
 

0
1
2

0
1
2
 
¹
	variables
trainable_variables
+metrics
,non_trainable_variables
-layer_regularization_losses

.layers
regularization_losses

/states
0layer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
trainable_variables
1metrics
2non_trainable_variables
3layer_regularization_losses

4layers
regularization_losses
5layer_metrics
RP
VARIABLE_VALUErnn_1/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE rnn_1/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUErnn_1/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

60
71
 
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

0
1
2

0
1
2
 
­
'	variables
(trainable_variables
8metrics
9non_trainable_variables
:layer_regularization_losses

;layers
)regularization_losses
<layer_metrics
 
 
 

0
 
 
 
 
 
 
 
4
	=total
	>count
?	variables
@	keras_api
D
	Atotal
	Bcount
C
_fn_kwargs
D	variables
E	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

D	variables

serving_default_masking_inputPlaceholder*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype0**
shape!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_masking_inputrnn_1/lstm_cell/kernelrnn_1/lstm_cell/bias rnn_1/lstm_cell/recurrent_kerneldense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_481943
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
â
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp*rnn_1/lstm_cell/kernel/Read/ReadVariableOp4rnn_1/lstm_cell/recurrent_kernel/Read/ReadVariableOp(rnn_1/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_484037
©
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasrnn_1/lstm_cell/kernel rnn_1/lstm_cell/recurrent_kernelrnn_1/lstm_cell/biastotalcounttotal_1count_1*
Tin
2
*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_484074öÃ
Ís
Ô
while_body_481658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstÄ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/ones_likep
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02&
$while/lstm_cell/split/ReadVariableOpë
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
while/lstm_cell/split¾
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÂ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1Â
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_2Â
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpß
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell/split_1³
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd¹
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_1¹
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_2¹
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_3
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul 
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_1 
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_2 
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_3«
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Ü
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_4«
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid¯
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2è
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_4¯
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2è
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_5
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_3¯
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2è
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh_1¢
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_6Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ý
­
+__inference_sequential_layer_call_fn_482584

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4818812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
«
Ã
while_cond_481657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481657___redundant_placeholder04
0while_while_cond_481657___redundant_placeholder14
0while_while_cond_481657___redundant_placeholder24
0while_while_cond_481657___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ò
´
+__inference_sequential_layer_call_fn_481926
masking_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallmasking_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4819132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'
_user_specified_namemasking_input
Ø

F__inference_sequential_layer_call_and_return_conditional_losses_481861
masking_input
rnn_1_481848
rnn_1_481850
rnn_1_481852
dense_481855
dense_481857
identity¢dense/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCallæ
masking/PartitionedCallPartitionedCallmasking_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_masking_layer_call_and_return_conditional_losses_4812352
masking/PartitionedCall¯
rnn_1/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0rnn_1_481848rnn_1_481850rnn_1_481852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4817862
rnn_1/StatefulPartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_481855dense_481857*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4818272
dense/StatefulPartitionedCallº
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:d `
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'
_user_specified_namemasking_input
ÅC
Î
A__inference_rnn_1_layer_call_and_return_conditional_losses_481080

inputs
lstm_cell_480999
lstm_cell_481001
lstm_cell_481003
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_480999lstm_cell_481001lstm_cell_481003*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_4806432#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_480999lstm_cell_481001lstm_cell_481003*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_481012*
condR
while_cond_481011*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
®
©
A__inference_dense_layer_call_and_return_conditional_losses_481827

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
)
Û
"__inference__traced_restore_484074
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias-
)assignvariableop_2_rnn_1_lstm_cell_kernel7
3assignvariableop_3_rnn_1_lstm_cell_recurrent_kernel+
'assignvariableop_4_rnn_1_lstm_cell_bias
assignvariableop_5_total
assignvariableop_6_count
assignvariableop_7_total_1
assignvariableop_8_count_1
identity_10¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8å
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*ñ
valueçBä
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¢
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slicesÝ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp)assignvariableop_2_rnn_1_lstm_cell_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¸
AssignVariableOp_3AssignVariableOp3assignvariableop_3_rnn_1_lstm_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_rnn_1_lstm_cell_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_totalIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_countIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_total_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp£

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9
Identity_10IdentityIdentity_9:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*
T0*
_output_shapes
: 2
Identity_10"#
identity_10Identity_10:output:0*9
_input_shapes(
&: :::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


&__inference_rnn_1_layer_call_fn_483181
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4812112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
«
Ã
while_cond_483596
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_483596___redundant_placeholder04
0while_while_cond_483596___redundant_placeholder14
0while_while_cond_483596___redundant_placeholder24
0while_while_cond_483596___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Àf

E__inference_lstm_cell_layer_call_and_return_conditional_losses_483876

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2Z
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeÓ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2¼¤Ë2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeØ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2îÜ+2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout_1/GreaterEqual/yÆ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeÙ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ÆÀ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout_2/GreaterEqual/yÆ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeØ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2½32(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout_3/GreaterEqual/yÆ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Mul_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È*
dtype02
split/ReadVariableOp«
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_6]
IdentityIdentity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identitya

Identity_1Identity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1
Â$
ü
while_body_481012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_481036_0
while_lstm_cell_481038_0
while_lstm_cell_481040_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_481036
while_lstm_cell_481038
while_lstm_cell_481040¢'while/lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÏ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_481036_0while_lstm_cell_481038_0while_lstm_cell_481040_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_4806432)
'while/lstm_cell/StatefulPartitionedCallô
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2·
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¾
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4¾
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_481036while_lstm_cell_481036_0"2
while_lstm_cell_481038while_lstm_cell_481038_0"2
while_lstm_cell_481040while_lstm_cell_481040_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Â$
ü
while_body_481143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_481167_0
while_lstm_cell_481169_0
while_lstm_cell_481171_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_481167
while_lstm_cell_481169
while_lstm_cell_481171¢'while/lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÏ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_481167_0while_lstm_cell_481169_0while_lstm_cell_481171_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_4807202)
'while/lstm_cell/StatefulPartitionedCallô
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2·
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¾
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4¾
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_481167while_lstm_cell_481167_0"2
while_lstm_cell_481169while_lstm_cell_481169_0"2
while_lstm_cell_481171while_lstm_cell_481171_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ã
ÿ
F__inference_sequential_layer_call_and_return_conditional_losses_481913

inputs
rnn_1_481900
rnn_1_481902
rnn_1_481904
dense_481907
dense_481909
identity¢dense/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCallß
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_masking_layer_call_and_return_conditional_losses_4812352
masking/PartitionedCall¯
rnn_1/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0rnn_1_481900rnn_1_481902rnn_1_481904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4817862
rnn_1/StatefulPartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_481907dense_481909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4818272
dense/StatefulPartitionedCallº
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¼
í
A__inference_rnn_1_layer_call_and_return_conditional_losses_483159
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const¬
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/ones_liked
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02 
lstm_cell/split/ReadVariableOpÓ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpÇ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd¡
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_1¡
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_2¡
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_3
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_1
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_2
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2¸
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_1
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_4
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_2o
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_5
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_2s
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh_1
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_483031*
condR
while_cond_483030*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1t
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
û£
Ô
while_body_483325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstÄ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/ones_like
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/dropout/Const¿
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Mul
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Ð26
4while/lstm_cell/dropout/random_uniform/RandomUniform
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2(
&while/lstm_cell/dropout/GreaterEqual/yþ
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$while/lstm_cell/dropout/GreaterEqual¯
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Castº
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Mul_1
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_1/ConstÅ
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_1/Mul
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ìÄ28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_1/GreaterEqual/y
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_1/GreaterEqualµ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_1/CastÂ
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_1/Mul_1
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_2/ConstÅ
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_2/Mul
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2îßL28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_2/GreaterEqual/y
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_2/GreaterEqualµ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_2/CastÂ
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_2/Mul_1
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_3/ConstÅ
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_3/Mul
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2±är28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_3/GreaterEqual/y
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_3/GreaterEqualµ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_3/CastÂ
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_3/Mul_1p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02&
$while/lstm_cell/split/ReadVariableOpë
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
while/lstm_cell/split¾
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÂ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1Â
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_2Â
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpß
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell/split_1³
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd¹
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_1¹
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_2¹
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_3
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul¡
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_1¡
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_2¡
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_3«
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Ü
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_4«
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid¯
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2è
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_4¯
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2è
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_5
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_3¯
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2è
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh_1¢
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_6Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
·
Ê
*__inference_lstm_cell_layer_call_fn_483987

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_4807202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1
ò
´
+__inference_sequential_layer_call_fn_481894
masking_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallmasking_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4818812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'
_user_specified_namemasking_input
Ã
ÿ
F__inference_sequential_layer_call_and_return_conditional_losses_481881

inputs
rnn_1_481868
rnn_1_481870
rnn_1_481872
dense_481875
dense_481877
identity¢dense/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCallß
masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_masking_layer_call_and_return_conditional_losses_4812352
masking/PartitionedCall¯
rnn_1/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0rnn_1_481868rnn_1_481870rnn_1_481872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4815462
rnn_1/StatefulPartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_481875dense_481877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4818272
dense/StatefulPartitionedCallº
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ís
Ô
while_body_483597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstÄ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/ones_likep
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02&
$while/lstm_cell/split/ReadVariableOpë
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
while/lstm_cell/split¾
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÂ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1Â
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_2Â
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpß
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell/split_1³
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd¹
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_1¹
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_2¹
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_3
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul 
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_1 
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_2 
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_3«
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Ü
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_4«
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid¯
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2è
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_4¯
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2è
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_5
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_3¯
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2è
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh_1¢
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_6Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Æ
­
$__inference_signature_wrapper_481943
masking_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallmasking_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_4804942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'
_user_specified_namemasking_input
áB

E__inference_lstm_cell_layer_call_and_return_conditional_losses_483953

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2Z
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	ones_likeP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È*
dtype02
split/ReadVariableOp«
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_6]
IdentityIdentity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identitya

Identity_1Identity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1
·
Ê
*__inference_lstm_cell_layer_call_fn_483970

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_4806432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1


&__inference_rnn_1_layer_call_fn_483747

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4817862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ÅC
Î
A__inference_rnn_1_layer_call_and_return_conditional_losses_481211

inputs
lstm_cell_481130
lstm_cell_481132
lstm_cell_481134
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_481130lstm_cell_481132lstm_cell_481134*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_4807202#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_481130lstm_cell_481132lstm_cell_481134*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_481143*
condR
while_cond_481142*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
§

rnn_1_while_cond_482102(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3
rnn_1_while_placeholder_4*
&rnn_1_while_less_rnn_1_strided_slice_1@
<rnn_1_while_rnn_1_while_cond_482102___redundant_placeholder0@
<rnn_1_while_rnn_1_while_cond_482102___redundant_placeholder1@
<rnn_1_while_rnn_1_while_cond_482102___redundant_placeholder2@
<rnn_1_while_rnn_1_while_cond_482102___redundant_placeholder3@
<rnn_1_while_rnn_1_while_cond_482102___redundant_placeholder4
rnn_1_while_identity

rnn_1/while/LessLessrnn_1_while_placeholder&rnn_1_while_less_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
rnn_1/while/Lesso
rnn_1/while/IdentityIdentityrnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn_1/while/Identity"5
rnn_1_while_identityrnn_1/while/Identity:output:0*j
_input_shapesY
W: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Î


rnn_1_while_body_482103(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3
rnn_1_while_placeholder_4'
#rnn_1_while_rnn_1_strided_slice_1_0c
_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0g
crnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_09
5rnn_1_while_lstm_cell_split_readvariableop_resource_0;
7rnn_1_while_lstm_cell_split_1_readvariableop_resource_03
/rnn_1_while_lstm_cell_readvariableop_resource_0
rnn_1_while_identity
rnn_1_while_identity_1
rnn_1_while_identity_2
rnn_1_while_identity_3
rnn_1_while_identity_4
rnn_1_while_identity_5
rnn_1_while_identity_6%
!rnn_1_while_rnn_1_strided_slice_1a
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensore
arnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor7
3rnn_1_while_lstm_cell_split_readvariableop_resource9
5rnn_1_while_lstm_cell_split_1_readvariableop_resource1
-rnn_1_while_lstm_cell_readvariableop_resourceÏ
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2?
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeø
/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0rnn_1_while_placeholderFrnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype021
/rnn_1/while/TensorArrayV2Read/TensorListGetItemÓ
?rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape
1rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemcrnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0rnn_1_while_placeholderHrnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
23
1rnn_1/while/TensorArrayV2Read_1/TensorListGetItem
%rnn_1/while/lstm_cell/ones_like/ShapeShapernn_1_while_placeholder_3*
T0*
_output_shapes
:2'
%rnn_1/while/lstm_cell/ones_like/Shape
%rnn_1/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%rnn_1/while/lstm_cell/ones_like/ConstÜ
rnn_1/while/lstm_cell/ones_likeFill.rnn_1/while/lstm_cell/ones_like/Shape:output:0.rnn_1/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/ones_like
#rnn_1/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#rnn_1/while/lstm_cell/dropout/Const×
!rnn_1/while/lstm_cell/dropout/MulMul(rnn_1/while/lstm_cell/ones_like:output:0,rnn_1/while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!rnn_1/while/lstm_cell/dropout/Mul¢
#rnn_1/while/lstm_cell/dropout/ShapeShape(rnn_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2%
#rnn_1/while/lstm_cell/dropout/Shape
:rnn_1/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform,rnn_1/while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Îâ2<
:rnn_1/while/lstm_cell/dropout/random_uniform/RandomUniform¡
,rnn_1/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2.
,rnn_1/while/lstm_cell/dropout/GreaterEqual/y
*rnn_1/while/lstm_cell/dropout/GreaterEqualGreaterEqualCrnn_1/while/lstm_cell/dropout/random_uniform/RandomUniform:output:05rnn_1/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*rnn_1/while/lstm_cell/dropout/GreaterEqualÁ
"rnn_1/while/lstm_cell/dropout/CastCast.rnn_1/while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"rnn_1/while/lstm_cell/dropout/CastÒ
#rnn_1/while/lstm_cell/dropout/Mul_1Mul%rnn_1/while/lstm_cell/dropout/Mul:z:0&rnn_1/while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#rnn_1/while/lstm_cell/dropout/Mul_1
%rnn_1/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%rnn_1/while/lstm_cell/dropout_1/ConstÝ
#rnn_1/while/lstm_cell/dropout_1/MulMul(rnn_1/while/lstm_cell/ones_like:output:0.rnn_1/while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#rnn_1/while/lstm_cell/dropout_1/Mul¦
%rnn_1/while/lstm_cell/dropout_1/ShapeShape(rnn_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2'
%rnn_1/while/lstm_cell/dropout_1/Shape
<rnn_1/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform.rnn_1/while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ìÛ²2>
<rnn_1/while/lstm_cell/dropout_1/random_uniform/RandomUniform¥
.rnn_1/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>20
.rnn_1/while/lstm_cell/dropout_1/GreaterEqual/y
,rnn_1/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualErnn_1/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:07rnn_1/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,rnn_1/while/lstm_cell/dropout_1/GreaterEqualÇ
$rnn_1/while/lstm_cell/dropout_1/CastCast0rnn_1/while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$rnn_1/while/lstm_cell/dropout_1/CastÚ
%rnn_1/while/lstm_cell/dropout_1/Mul_1Mul'rnn_1/while/lstm_cell/dropout_1/Mul:z:0(rnn_1/while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%rnn_1/while/lstm_cell/dropout_1/Mul_1
%rnn_1/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%rnn_1/while/lstm_cell/dropout_2/ConstÝ
#rnn_1/while/lstm_cell/dropout_2/MulMul(rnn_1/while/lstm_cell/ones_like:output:0.rnn_1/while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#rnn_1/while/lstm_cell/dropout_2/Mul¦
%rnn_1/while/lstm_cell/dropout_2/ShapeShape(rnn_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2'
%rnn_1/while/lstm_cell/dropout_2/Shape
<rnn_1/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform.rnn_1/while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2¹¾ë2>
<rnn_1/while/lstm_cell/dropout_2/random_uniform/RandomUniform¥
.rnn_1/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>20
.rnn_1/while/lstm_cell/dropout_2/GreaterEqual/y
,rnn_1/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualErnn_1/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:07rnn_1/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,rnn_1/while/lstm_cell/dropout_2/GreaterEqualÇ
$rnn_1/while/lstm_cell/dropout_2/CastCast0rnn_1/while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$rnn_1/while/lstm_cell/dropout_2/CastÚ
%rnn_1/while/lstm_cell/dropout_2/Mul_1Mul'rnn_1/while/lstm_cell/dropout_2/Mul:z:0(rnn_1/while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%rnn_1/while/lstm_cell/dropout_2/Mul_1
%rnn_1/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%rnn_1/while/lstm_cell/dropout_3/ConstÝ
#rnn_1/while/lstm_cell/dropout_3/MulMul(rnn_1/while/lstm_cell/ones_like:output:0.rnn_1/while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#rnn_1/while/lstm_cell/dropout_3/Mul¦
%rnn_1/while/lstm_cell/dropout_3/ShapeShape(rnn_1/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2'
%rnn_1/while/lstm_cell/dropout_3/Shape
<rnn_1/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform.rnn_1/while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2µ¦2>
<rnn_1/while/lstm_cell/dropout_3/random_uniform/RandomUniform¥
.rnn_1/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>20
.rnn_1/while/lstm_cell/dropout_3/GreaterEqual/y
,rnn_1/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualErnn_1/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:07rnn_1/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,rnn_1/while/lstm_cell/dropout_3/GreaterEqualÇ
$rnn_1/while/lstm_cell/dropout_3/CastCast0rnn_1/while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$rnn_1/while/lstm_cell/dropout_3/CastÚ
%rnn_1/while/lstm_cell/dropout_3/Mul_1Mul'rnn_1/while/lstm_cell/dropout_3/Mul:z:0(rnn_1/while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%rnn_1/while/lstm_cell/dropout_3/Mul_1|
rnn_1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/lstm_cell/Const
%rnn_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%rnn_1/while/lstm_cell/split/split_dimÐ
*rnn_1/while/lstm_cell/split/ReadVariableOpReadVariableOp5rnn_1_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02,
*rnn_1/while/lstm_cell/split/ReadVariableOp
rnn_1/while/lstm_cell/splitSplit.rnn_1/while/lstm_cell/split/split_dim:output:02rnn_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
rnn_1/while/lstm_cell/splitÖ
rnn_1/while/lstm_cell/MatMulMatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/MatMulÚ
rnn_1/while/lstm_cell/MatMul_1MatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_1Ú
rnn_1/while/lstm_cell/MatMul_2MatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_2Ú
rnn_1/while/lstm_cell/MatMul_3MatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_3
rnn_1/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/lstm_cell/Const_1
'rnn_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'rnn_1/while/lstm_cell/split_1/split_dimÑ
,rnn_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp7rnn_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02.
,rnn_1/while/lstm_cell/split_1/ReadVariableOp÷
rnn_1/while/lstm_cell/split_1Split0rnn_1/while/lstm_cell/split_1/split_dim:output:04rnn_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
rnn_1/while/lstm_cell/split_1Ë
rnn_1/while/lstm_cell/BiasAddBiasAdd&rnn_1/while/lstm_cell/MatMul:product:0&rnn_1/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/BiasAddÑ
rnn_1/while/lstm_cell/BiasAdd_1BiasAdd(rnn_1/while/lstm_cell/MatMul_1:product:0&rnn_1/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/BiasAdd_1Ñ
rnn_1/while/lstm_cell/BiasAdd_2BiasAdd(rnn_1/while/lstm_cell/MatMul_2:product:0&rnn_1/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/BiasAdd_2Ñ
rnn_1/while/lstm_cell/BiasAdd_3BiasAdd(rnn_1/while/lstm_cell/MatMul_3:product:0&rnn_1/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/BiasAdd_3³
rnn_1/while/lstm_cell/mulMulrnn_1_while_placeholder_3'rnn_1/while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul¹
rnn_1/while/lstm_cell/mul_1Mulrnn_1_while_placeholder_3)rnn_1/while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_1¹
rnn_1/while/lstm_cell/mul_2Mulrnn_1_while_placeholder_3)rnn_1/while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_2¹
rnn_1/while/lstm_cell/mul_3Mulrnn_1_while_placeholder_3)rnn_1/while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_3½
$rnn_1/while/lstm_cell/ReadVariableOpReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02&
$rnn_1/while/lstm_cell/ReadVariableOp§
)rnn_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)rnn_1/while/lstm_cell/strided_slice/stack«
+rnn_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+rnn_1/while/lstm_cell/strided_slice/stack_1«
+rnn_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+rnn_1/while/lstm_cell/strided_slice/stack_2
#rnn_1/while/lstm_cell/strided_sliceStridedSlice,rnn_1/while/lstm_cell/ReadVariableOp:value:02rnn_1/while/lstm_cell/strided_slice/stack:output:04rnn_1/while/lstm_cell/strided_slice/stack_1:output:04rnn_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2%
#rnn_1/while/lstm_cell/strided_sliceÉ
rnn_1/while/lstm_cell/MatMul_4MatMulrnn_1/while/lstm_cell/mul:z:0,rnn_1/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_4Ã
rnn_1/while/lstm_cell/addAddV2&rnn_1/while/lstm_cell/BiasAdd:output:0(rnn_1/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add
rnn_1/while/lstm_cell/SigmoidSigmoidrnn_1/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/SigmoidÁ
&rnn_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02(
&rnn_1/while/lstm_cell/ReadVariableOp_1«
+rnn_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2-
+rnn_1/while/lstm_cell/strided_slice_1/stack¯
-rnn_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-rnn_1/while/lstm_cell/strided_slice_1/stack_1¯
-rnn_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-rnn_1/while/lstm_cell/strided_slice_1/stack_2
%rnn_1/while/lstm_cell/strided_slice_1StridedSlice.rnn_1/while/lstm_cell/ReadVariableOp_1:value:04rnn_1/while/lstm_cell/strided_slice_1/stack:output:06rnn_1/while/lstm_cell/strided_slice_1/stack_1:output:06rnn_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2'
%rnn_1/while/lstm_cell/strided_slice_1Í
rnn_1/while/lstm_cell/MatMul_5MatMulrnn_1/while/lstm_cell/mul_1:z:0.rnn_1/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_5É
rnn_1/while/lstm_cell/add_1AddV2(rnn_1/while/lstm_cell/BiasAdd_1:output:0(rnn_1/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_1 
rnn_1/while/lstm_cell/Sigmoid_1Sigmoidrnn_1/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/Sigmoid_1³
rnn_1/while/lstm_cell/mul_4Mul#rnn_1/while/lstm_cell/Sigmoid_1:y:0rnn_1_while_placeholder_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_4Á
&rnn_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02(
&rnn_1/while/lstm_cell/ReadVariableOp_2«
+rnn_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+rnn_1/while/lstm_cell/strided_slice_2/stack¯
-rnn_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2/
-rnn_1/while/lstm_cell/strided_slice_2/stack_1¯
-rnn_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-rnn_1/while/lstm_cell/strided_slice_2/stack_2
%rnn_1/while/lstm_cell/strided_slice_2StridedSlice.rnn_1/while/lstm_cell/ReadVariableOp_2:value:04rnn_1/while/lstm_cell/strided_slice_2/stack:output:06rnn_1/while/lstm_cell/strided_slice_2/stack_1:output:06rnn_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2'
%rnn_1/while/lstm_cell/strided_slice_2Í
rnn_1/while/lstm_cell/MatMul_6MatMulrnn_1/while/lstm_cell/mul_2:z:0.rnn_1/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_6É
rnn_1/while/lstm_cell/add_2AddV2(rnn_1/while/lstm_cell/BiasAdd_2:output:0(rnn_1/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_2
rnn_1/while/lstm_cell/TanhTanhrnn_1/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/Tanh¶
rnn_1/while/lstm_cell/mul_5Mul!rnn_1/while/lstm_cell/Sigmoid:y:0rnn_1/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_5·
rnn_1/while/lstm_cell/add_3AddV2rnn_1/while/lstm_cell/mul_4:z:0rnn_1/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_3Á
&rnn_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02(
&rnn_1/while/lstm_cell/ReadVariableOp_3«
+rnn_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2-
+rnn_1/while/lstm_cell/strided_slice_3/stack¯
-rnn_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-rnn_1/while/lstm_cell/strided_slice_3/stack_1¯
-rnn_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-rnn_1/while/lstm_cell/strided_slice_3/stack_2
%rnn_1/while/lstm_cell/strided_slice_3StridedSlice.rnn_1/while/lstm_cell/ReadVariableOp_3:value:04rnn_1/while/lstm_cell/strided_slice_3/stack:output:06rnn_1/while/lstm_cell/strided_slice_3/stack_1:output:06rnn_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2'
%rnn_1/while/lstm_cell/strided_slice_3Í
rnn_1/while/lstm_cell/MatMul_7MatMulrnn_1/while/lstm_cell/mul_3:z:0.rnn_1/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_7É
rnn_1/while/lstm_cell/add_4AddV2(rnn_1/while/lstm_cell/BiasAdd_3:output:0(rnn_1/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_4 
rnn_1/while/lstm_cell/Sigmoid_2Sigmoidrnn_1/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/Sigmoid_2
rnn_1/while/lstm_cell/Tanh_1Tanhrnn_1/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/Tanh_1º
rnn_1/while/lstm_cell/mul_6Mul#rnn_1/while/lstm_cell/Sigmoid_2:y:0 rnn_1/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_6
rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
rnn_1/while/Tile/multiples½
rnn_1/while/TileTile8rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0#rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn_1/while/TileÁ
rnn_1/while/SelectV2SelectV2rnn_1/while/Tile:output:0rnn_1/while/lstm_cell/mul_6:z:0rnn_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/SelectV2
rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
rnn_1/while/Tile_1/multiplesÃ
rnn_1/while/Tile_1Tile8rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0%rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn_1/while/Tile_1
rnn_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
rnn_1/while/Tile_2/multiplesÃ
rnn_1/while/Tile_2Tile8rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0%rnn_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn_1/while/Tile_2Ç
rnn_1/while/SelectV2_1SelectV2rnn_1/while/Tile_1:output:0rnn_1/while/lstm_cell/mul_6:z:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/SelectV2_1Ç
rnn_1/while/SelectV2_2SelectV2rnn_1/while/Tile_2:output:0rnn_1/while/lstm_cell/add_3:z:0rnn_1_while_placeholder_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/SelectV2_2ù
0rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_1_while_placeholder_1rnn_1_while_placeholderrnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype022
0rnn_1/while/TensorArrayV2Write/TensorListSetItemh
rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add/y
rnn_1/while/addAddV2rnn_1_while_placeholderrnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/addl
rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add_1/y
rnn_1/while/add_1AddV2$rnn_1_while_rnn_1_while_loop_counterrnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/add_1p
rnn_1/while/IdentityIdentityrnn_1/while/add_1:z:0*
T0*
_output_shapes
: 2
rnn_1/while/Identity
rnn_1/while/Identity_1Identity*rnn_1_while_rnn_1_while_maximum_iterations*
T0*
_output_shapes
: 2
rnn_1/while/Identity_1r
rnn_1/while/Identity_2Identityrnn_1/while/add:z:0*
T0*
_output_shapes
: 2
rnn_1/while/Identity_2
rnn_1/while/Identity_3Identity@rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rnn_1/while/Identity_3
rnn_1/while/Identity_4Identityrnn_1/while/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/Identity_4
rnn_1/while/Identity_5Identityrnn_1/while/SelectV2_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/Identity_5
rnn_1/while/Identity_6Identityrnn_1/while/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/Identity_6"5
rnn_1_while_identityrnn_1/while/Identity:output:0"9
rnn_1_while_identity_1rnn_1/while/Identity_1:output:0"9
rnn_1_while_identity_2rnn_1/while/Identity_2:output:0"9
rnn_1_while_identity_3rnn_1/while/Identity_3:output:0"9
rnn_1_while_identity_4rnn_1/while/Identity_4:output:0"9
rnn_1_while_identity_5rnn_1/while/Identity_5:output:0"9
rnn_1_while_identity_6rnn_1/while/Identity_6:output:0"`
-rnn_1_while_lstm_cell_readvariableop_resource/rnn_1_while_lstm_cell_readvariableop_resource_0"p
5rnn_1_while_lstm_cell_split_1_readvariableop_resource7rnn_1_while_lstm_cell_split_1_readvariableop_resource_0"l
3rnn_1_while_lstm_cell_split_readvariableop_resource5rnn_1_while_lstm_cell_split_readvariableop_resource_0"H
!rnn_1_while_rnn_1_strided_slice_1#rnn_1_while_rnn_1_strided_slice_1_0"È
arnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensorcrnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"À
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
«
Ã
while_cond_481011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481011___redundant_placeholder04
0while_while_cond_481011___redundant_placeholder14
0while_while_cond_481011___redundant_placeholder24
0while_while_cond_481011___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ü£
Ô
while_body_481386
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstÄ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/ones_like
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/dropout/Const¿
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Mul
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Å26
4while/lstm_cell/dropout/random_uniform/RandomUniform
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2(
&while/lstm_cell/dropout/GreaterEqual/yþ
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$while/lstm_cell/dropout/GreaterEqual¯
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Castº
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Mul_1
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_1/ConstÅ
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_1/Mul
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2· 28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_1/GreaterEqual/y
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_1/GreaterEqualµ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_1/CastÂ
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_1/Mul_1
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_2/ConstÅ
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_2/Mul
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ñ28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_2/GreaterEqual/y
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_2/GreaterEqualµ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_2/CastÂ
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_2/Mul_1
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_3/ConstÅ
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_3/Mul
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2çÙ^28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_3/GreaterEqual/y
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_3/GreaterEqualµ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_3/CastÂ
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_3/Mul_1p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02&
$while/lstm_cell/split/ReadVariableOpë
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
while/lstm_cell/split¾
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÂ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1Â
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_2Â
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpß
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell/split_1³
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd¹
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_1¹
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_2¹
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_3
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul¡
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_1¡
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_2¡
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_3«
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Ü
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_4«
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid¯
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2è
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_4¯
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2è
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_5
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_3¯
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2è
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh_1¢
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_6Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ö
{
&__inference_dense_layer_call_fn_483767

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4818272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


"sequential_rnn_1_while_cond_480340>
:sequential_rnn_1_while_sequential_rnn_1_while_loop_counterD
@sequential_rnn_1_while_sequential_rnn_1_while_maximum_iterations&
"sequential_rnn_1_while_placeholder(
$sequential_rnn_1_while_placeholder_1(
$sequential_rnn_1_while_placeholder_2(
$sequential_rnn_1_while_placeholder_3(
$sequential_rnn_1_while_placeholder_4@
<sequential_rnn_1_while_less_sequential_rnn_1_strided_slice_1V
Rsequential_rnn_1_while_sequential_rnn_1_while_cond_480340___redundant_placeholder0V
Rsequential_rnn_1_while_sequential_rnn_1_while_cond_480340___redundant_placeholder1V
Rsequential_rnn_1_while_sequential_rnn_1_while_cond_480340___redundant_placeholder2V
Rsequential_rnn_1_while_sequential_rnn_1_while_cond_480340___redundant_placeholder3V
Rsequential_rnn_1_while_sequential_rnn_1_while_cond_480340___redundant_placeholder4#
sequential_rnn_1_while_identity
Å
sequential/rnn_1/while/LessLess"sequential_rnn_1_while_placeholder<sequential_rnn_1_while_less_sequential_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
sequential/rnn_1/while/Less
sequential/rnn_1/while/IdentityIdentitysequential/rnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2!
sequential/rnn_1/while/Identity"K
sequential_rnn_1_while_identity(sequential/rnn_1/while/Identity:output:0*j
_input_shapesY
W: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ø

F__inference_sequential_layer_call_and_return_conditional_losses_481844
masking_input
rnn_1_481809
rnn_1_481811
rnn_1_481813
dense_481838
dense_481840
identity¢dense/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCallæ
masking/PartitionedCallPartitionedCallmasking_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_masking_layer_call_and_return_conditional_losses_4812352
masking/PartitionedCall¯
rnn_1/StatefulPartitionedCallStatefulPartitionedCall masking/PartitionedCall:output:0rnn_1_481809rnn_1_481811rnn_1_481813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4815462
rnn_1/StatefulPartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_481838dense_481840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4818272
dense/StatefulPartitionedCallº
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:d `
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'
_user_specified_namemasking_input
«
Ã
while_cond_482758
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_482758___redundant_placeholder04
0while_while_cond_482758___redundant_placeholder14
0while_while_cond_482758___redundant_placeholder24
0while_while_cond_482758___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Â	
_
C__inference_masking_layer_call_and_return_conditional_losses_482610

inputs
identity]

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2

NotEqual/y}
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Any/reduction_indices
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
mul
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¶
ë
A__inference_rnn_1_layer_call_and_return_conditional_losses_481546

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const¬
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout/Const§
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeñ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2³ß20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 lstm_cell/dropout/GreaterEqual/yæ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Cast¢
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeö
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Í]22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_1/GreaterEqual/yî
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_1/GreaterEqual£
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Castª
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape÷
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2®ÄÄ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_2/GreaterEqual/yî
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_2/GreaterEqual£
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Castª
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeö
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Ë122
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_3/GreaterEqual/yî
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_3/GreaterEqual£
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Castª
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Mul_1d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02 
lstm_cell/split/ReadVariableOpÓ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpÇ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd¡
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_1¡
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_2¡
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_3
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_1
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_2
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2¸
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_1
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_4
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_2o
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_5
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_2s
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh_1
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_481386*
condR
while_cond_481385*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1t
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¶
ë
A__inference_rnn_1_layer_call_and_return_conditional_losses_483485

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const¬
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout/Const§
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeð
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ª¶20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 lstm_cell/dropout/GreaterEqual/yæ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Cast¢
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shape÷
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2°®ì22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_1/GreaterEqual/yî
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_1/GreaterEqual£
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Castª
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape÷
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2¢½22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_2/GreaterEqual/yî
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_2/GreaterEqual£
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Castª
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape÷
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2÷22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_3/GreaterEqual/yî
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_3/GreaterEqual£
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Castª
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Mul_1d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02 
lstm_cell/split/ReadVariableOpÓ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpÇ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd¡
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_1¡
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_2¡
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_3
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_1
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_2
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2¸
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_1
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_4
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_2o
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_5
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_2s
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh_1
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_483325*
condR
while_cond_483324*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1t
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
´
ë
A__inference_rnn_1_layer_call_and_return_conditional_losses_481786

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const¬
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/ones_liked
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02 
lstm_cell/split/ReadVariableOpÓ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpÇ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd¡
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_1¡
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_2¡
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_3
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_1
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_2
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2¸
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_1
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_4
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_2o
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_5
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_2s
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh_1
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_481658*
condR
while_cond_481657*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1t
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ÑB

E__inference_lstm_cell_layer_call_and_return_conditional_losses_480720

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	ones_likeP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È*
dtype02
split/ReadVariableOp«
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_3_
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mulc
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_1c
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_2c
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_6]
IdentityIdentity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identitya

Identity_1Identity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
è
Ý
F__inference_sequential_layer_call_and_return_conditional_losses_482288

inputs1
-rnn_1_lstm_cell_split_readvariableop_resource3
/rnn_1_lstm_cell_split_1_readvariableop_resource+
'rnn_1_lstm_cell_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢rnn_1/whilem
masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking/NotEqual/y
masking/NotEqualNotEqualinputsmasking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
masking/NotEqual
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
masking/Any/reduction_indices¦
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
masking/Any
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
masking/mul
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
masking/SqueezeY
rnn_1/ShapeShapemasking/mul:z:0*
T0*
_output_shapes
:2
rnn_1/Shape
rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice/stack
rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_1
rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_2
rnn_1/strided_sliceStridedSlicernn_1/Shape:output:0"rnn_1/strided_slice/stack:output:0$rnn_1/strided_slice/stack_1:output:0$rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_sliceh
rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros/mul/y
rnn_1/zeros/mulMulrnn_1/strided_slice:output:0rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/mulk
rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros/Less/y
rnn_1/zeros/LessLessrnn_1/zeros/mul:z:0rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/Lessn
rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros/packed/1
rnn_1/zeros/packedPackrnn_1/strided_slice:output:0rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros/packedk
rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros/Const
rnn_1/zerosFillrnn_1/zeros/packed:output:0rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/zerosl
rnn_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros_1/mul/y
rnn_1/zeros_1/mulMulrnn_1/strided_slice:output:0rnn_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/mulo
rnn_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros_1/Less/y
rnn_1/zeros_1/LessLessrnn_1/zeros_1/mul:z:0rnn_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/Lessr
rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros_1/packed/1¡
rnn_1/zeros_1/packedPackrnn_1/strided_slice:output:0rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros_1/packedo
rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros_1/Const
rnn_1/zeros_1Fillrnn_1/zeros_1/packed:output:0rnn_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/zeros_1
rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose/perm
rnn_1/transpose	Transposemasking/mul:z:0rnn_1/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
rnn_1/transposea
rnn_1/Shape_1Shapernn_1/transpose:y:0*
T0*
_output_shapes
:2
rnn_1/Shape_1
rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_1/stack
rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_1
rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_2
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1:output:0$rnn_1/strided_slice_1/stack:output:0&rnn_1/strided_slice_1/stack_1:output:0&rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_slice_1w
rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rnn_1/ExpandDims/dimª
rnn_1/ExpandDims
ExpandDimsmasking/Squeeze:output:0rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn_1/ExpandDims
rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose_1/perm®
rnn_1/transpose_1	Transposernn_1/ExpandDims:output:0rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn_1/transpose_1
!rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!rnn_1/TensorArrayV2/element_shapeÊ
rnn_1/TensorArrayV2TensorListReserve*rnn_1/TensorArrayV2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2Ë
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2=
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_1/transpose:y:0Drnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-rnn_1/TensorArrayUnstack/TensorListFromTensor
rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_2/stack
rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_1
rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_2¡
rnn_1/strided_slice_2StridedSlicernn_1/transpose:y:0$rnn_1/strided_slice_2/stack:output:0&rnn_1/strided_slice_2/stack_1:output:0&rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
rnn_1/strided_slice_2
rnn_1/lstm_cell/ones_like/ShapeShapernn_1/zeros:output:0*
T0*
_output_shapes
:2!
rnn_1/lstm_cell/ones_like/Shape
rnn_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
rnn_1/lstm_cell/ones_like/ConstÄ
rnn_1/lstm_cell/ones_likeFill(rnn_1/lstm_cell/ones_like/Shape:output:0(rnn_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/ones_like
rnn_1/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
rnn_1/lstm_cell/dropout/Const¿
rnn_1/lstm_cell/dropout/MulMul"rnn_1/lstm_cell/ones_like:output:0&rnn_1/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/dropout/Mul
rnn_1/lstm_cell/dropout/ShapeShape"rnn_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
rnn_1/lstm_cell/dropout/Shape
4rnn_1/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&rnn_1/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Â26
4rnn_1/lstm_cell/dropout/random_uniform/RandomUniform
&rnn_1/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2(
&rnn_1/lstm_cell/dropout/GreaterEqual/yþ
$rnn_1/lstm_cell/dropout/GreaterEqualGreaterEqual=rnn_1/lstm_cell/dropout/random_uniform/RandomUniform:output:0/rnn_1/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$rnn_1/lstm_cell/dropout/GreaterEqual¯
rnn_1/lstm_cell/dropout/CastCast(rnn_1/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/dropout/Castº
rnn_1/lstm_cell/dropout/Mul_1Mulrnn_1/lstm_cell/dropout/Mul:z:0 rnn_1/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/dropout/Mul_1
rnn_1/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
rnn_1/lstm_cell/dropout_1/ConstÅ
rnn_1/lstm_cell/dropout_1/MulMul"rnn_1/lstm_cell/ones_like:output:0(rnn_1/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/dropout_1/Mul
rnn_1/lstm_cell/dropout_1/ShapeShape"rnn_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
rnn_1/lstm_cell/dropout_1/Shape
6rnn_1/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(rnn_1/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2¾åÖ28
6rnn_1/lstm_cell/dropout_1/random_uniform/RandomUniform
(rnn_1/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(rnn_1/lstm_cell/dropout_1/GreaterEqual/y
&rnn_1/lstm_cell/dropout_1/GreaterEqualGreaterEqual?rnn_1/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01rnn_1/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&rnn_1/lstm_cell/dropout_1/GreaterEqualµ
rnn_1/lstm_cell/dropout_1/CastCast*rnn_1/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/lstm_cell/dropout_1/CastÂ
rnn_1/lstm_cell/dropout_1/Mul_1Mul!rnn_1/lstm_cell/dropout_1/Mul:z:0"rnn_1/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/lstm_cell/dropout_1/Mul_1
rnn_1/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
rnn_1/lstm_cell/dropout_2/ConstÅ
rnn_1/lstm_cell/dropout_2/MulMul"rnn_1/lstm_cell/ones_like:output:0(rnn_1/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/dropout_2/Mul
rnn_1/lstm_cell/dropout_2/ShapeShape"rnn_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
rnn_1/lstm_cell/dropout_2/Shape
6rnn_1/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(rnn_1/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Ðõ28
6rnn_1/lstm_cell/dropout_2/random_uniform/RandomUniform
(rnn_1/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(rnn_1/lstm_cell/dropout_2/GreaterEqual/y
&rnn_1/lstm_cell/dropout_2/GreaterEqualGreaterEqual?rnn_1/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01rnn_1/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&rnn_1/lstm_cell/dropout_2/GreaterEqualµ
rnn_1/lstm_cell/dropout_2/CastCast*rnn_1/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/lstm_cell/dropout_2/CastÂ
rnn_1/lstm_cell/dropout_2/Mul_1Mul!rnn_1/lstm_cell/dropout_2/Mul:z:0"rnn_1/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/lstm_cell/dropout_2/Mul_1
rnn_1/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
rnn_1/lstm_cell/dropout_3/ConstÅ
rnn_1/lstm_cell/dropout_3/MulMul"rnn_1/lstm_cell/ones_like:output:0(rnn_1/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/dropout_3/Mul
rnn_1/lstm_cell/dropout_3/ShapeShape"rnn_1/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
rnn_1/lstm_cell/dropout_3/Shape
6rnn_1/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(rnn_1/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ðß28
6rnn_1/lstm_cell/dropout_3/random_uniform/RandomUniform
(rnn_1/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(rnn_1/lstm_cell/dropout_3/GreaterEqual/y
&rnn_1/lstm_cell/dropout_3/GreaterEqualGreaterEqual?rnn_1/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01rnn_1/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&rnn_1/lstm_cell/dropout_3/GreaterEqualµ
rnn_1/lstm_cell/dropout_3/CastCast*rnn_1/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/lstm_cell/dropout_3/CastÂ
rnn_1/lstm_cell/dropout_3/Mul_1Mul!rnn_1/lstm_cell/dropout_3/Mul:z:0"rnn_1/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/lstm_cell/dropout_3/Mul_1p
rnn_1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/lstm_cell/Const
rnn_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
rnn_1/lstm_cell/split/split_dim¼
$rnn_1/lstm_cell/split/ReadVariableOpReadVariableOp-rnn_1_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02&
$rnn_1/lstm_cell/split/ReadVariableOpë
rnn_1/lstm_cell/splitSplit(rnn_1/lstm_cell/split/split_dim:output:0,rnn_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
rnn_1/lstm_cell/split¬
rnn_1/lstm_cell/MatMulMatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul°
rnn_1/lstm_cell/MatMul_1MatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_1°
rnn_1/lstm_cell/MatMul_2MatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_2°
rnn_1/lstm_cell/MatMul_3MatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_3t
rnn_1/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/lstm_cell/Const_1
!rnn_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!rnn_1/lstm_cell/split_1/split_dim½
&rnn_1/lstm_cell/split_1/ReadVariableOpReadVariableOp/rnn_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02(
&rnn_1/lstm_cell/split_1/ReadVariableOpß
rnn_1/lstm_cell/split_1Split*rnn_1/lstm_cell/split_1/split_dim:output:0.rnn_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
rnn_1/lstm_cell/split_1³
rnn_1/lstm_cell/BiasAddBiasAdd rnn_1/lstm_cell/MatMul:product:0 rnn_1/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd¹
rnn_1/lstm_cell/BiasAdd_1BiasAdd"rnn_1/lstm_cell/MatMul_1:product:0 rnn_1/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd_1¹
rnn_1/lstm_cell/BiasAdd_2BiasAdd"rnn_1/lstm_cell/MatMul_2:product:0 rnn_1/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd_2¹
rnn_1/lstm_cell/BiasAdd_3BiasAdd"rnn_1/lstm_cell/MatMul_3:product:0 rnn_1/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd_3
rnn_1/lstm_cell/mulMulrnn_1/zeros:output:0!rnn_1/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul¢
rnn_1/lstm_cell/mul_1Mulrnn_1/zeros:output:0#rnn_1/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_1¢
rnn_1/lstm_cell/mul_2Mulrnn_1/zeros:output:0#rnn_1/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_2¢
rnn_1/lstm_cell/mul_3Mulrnn_1/zeros:output:0#rnn_1/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_3©
rnn_1/lstm_cell/ReadVariableOpReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02 
rnn_1/lstm_cell/ReadVariableOp
#rnn_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#rnn_1/lstm_cell/strided_slice/stack
%rnn_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%rnn_1/lstm_cell/strided_slice/stack_1
%rnn_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%rnn_1/lstm_cell/strided_slice/stack_2Ü
rnn_1/lstm_cell/strided_sliceStridedSlice&rnn_1/lstm_cell/ReadVariableOp:value:0,rnn_1/lstm_cell/strided_slice/stack:output:0.rnn_1/lstm_cell/strided_slice/stack_1:output:0.rnn_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
rnn_1/lstm_cell/strided_slice±
rnn_1/lstm_cell/MatMul_4MatMulrnn_1/lstm_cell/mul:z:0&rnn_1/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_4«
rnn_1/lstm_cell/addAddV2 rnn_1/lstm_cell/BiasAdd:output:0"rnn_1/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add
rnn_1/lstm_cell/SigmoidSigmoidrnn_1/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Sigmoid­
 rnn_1/lstm_cell/ReadVariableOp_1ReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 rnn_1/lstm_cell/ReadVariableOp_1
%rnn_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%rnn_1/lstm_cell/strided_slice_1/stack£
'rnn_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'rnn_1/lstm_cell/strided_slice_1/stack_1£
'rnn_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rnn_1/lstm_cell/strided_slice_1/stack_2è
rnn_1/lstm_cell/strided_slice_1StridedSlice(rnn_1/lstm_cell/ReadVariableOp_1:value:0.rnn_1/lstm_cell/strided_slice_1/stack:output:00rnn_1/lstm_cell/strided_slice_1/stack_1:output:00rnn_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
rnn_1/lstm_cell/strided_slice_1µ
rnn_1/lstm_cell/MatMul_5MatMulrnn_1/lstm_cell/mul_1:z:0(rnn_1/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_5±
rnn_1/lstm_cell/add_1AddV2"rnn_1/lstm_cell/BiasAdd_1:output:0"rnn_1/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_1
rnn_1/lstm_cell/Sigmoid_1Sigmoidrnn_1/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Sigmoid_1
rnn_1/lstm_cell/mul_4Mulrnn_1/lstm_cell/Sigmoid_1:y:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_4­
 rnn_1/lstm_cell/ReadVariableOp_2ReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 rnn_1/lstm_cell/ReadVariableOp_2
%rnn_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%rnn_1/lstm_cell/strided_slice_2/stack£
'rnn_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'rnn_1/lstm_cell/strided_slice_2/stack_1£
'rnn_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rnn_1/lstm_cell/strided_slice_2/stack_2è
rnn_1/lstm_cell/strided_slice_2StridedSlice(rnn_1/lstm_cell/ReadVariableOp_2:value:0.rnn_1/lstm_cell/strided_slice_2/stack:output:00rnn_1/lstm_cell/strided_slice_2/stack_1:output:00rnn_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
rnn_1/lstm_cell/strided_slice_2µ
rnn_1/lstm_cell/MatMul_6MatMulrnn_1/lstm_cell/mul_2:z:0(rnn_1/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_6±
rnn_1/lstm_cell/add_2AddV2"rnn_1/lstm_cell/BiasAdd_2:output:0"rnn_1/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_2
rnn_1/lstm_cell/TanhTanhrnn_1/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Tanh
rnn_1/lstm_cell/mul_5Mulrnn_1/lstm_cell/Sigmoid:y:0rnn_1/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_5
rnn_1/lstm_cell/add_3AddV2rnn_1/lstm_cell/mul_4:z:0rnn_1/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_3­
 rnn_1/lstm_cell/ReadVariableOp_3ReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 rnn_1/lstm_cell/ReadVariableOp_3
%rnn_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%rnn_1/lstm_cell/strided_slice_3/stack£
'rnn_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rnn_1/lstm_cell/strided_slice_3/stack_1£
'rnn_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rnn_1/lstm_cell/strided_slice_3/stack_2è
rnn_1/lstm_cell/strided_slice_3StridedSlice(rnn_1/lstm_cell/ReadVariableOp_3:value:0.rnn_1/lstm_cell/strided_slice_3/stack:output:00rnn_1/lstm_cell/strided_slice_3/stack_1:output:00rnn_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
rnn_1/lstm_cell/strided_slice_3µ
rnn_1/lstm_cell/MatMul_7MatMulrnn_1/lstm_cell/mul_3:z:0(rnn_1/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_7±
rnn_1/lstm_cell/add_4AddV2"rnn_1/lstm_cell/BiasAdd_3:output:0"rnn_1/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_4
rnn_1/lstm_cell/Sigmoid_2Sigmoidrnn_1/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Sigmoid_2
rnn_1/lstm_cell/Tanh_1Tanhrnn_1/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Tanh_1¢
rnn_1/lstm_cell/mul_6Mulrnn_1/lstm_cell/Sigmoid_2:y:0rnn_1/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_6
#rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2%
#rnn_1/TensorArrayV2_1/element_shapeÐ
rnn_1/TensorArrayV2_1TensorListReserve,rnn_1/TensorArrayV2_1/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2_1Z

rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn_1/time
#rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#rnn_1/TensorArrayV2_2/element_shapeÐ
rnn_1/TensorArrayV2_2TensorListReserve,rnn_1/TensorArrayV2_2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
rnn_1/TensorArrayV2_2Ï
=rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape
/rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorrnn_1/transpose_1:y:0Frnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type021
/rnn_1/TensorArrayUnstack_1/TensorListFromTensor~
rnn_1/zeros_like	ZerosLikernn_1/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/zeros_like
rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
rnn_1/while/maximum_iterationsv
rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/while/loop_counter¸
rnn_1/whileWhile!rnn_1/while/loop_counter:output:0'rnn_1/while/maximum_iterations:output:0rnn_1/time:output:0rnn_1/TensorArrayV2_1:handle:0rnn_1/zeros_like:y:0rnn_1/zeros:output:0rnn_1/zeros_1:output:0rnn_1/strided_slice_1:output:0=rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0?rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0-rnn_1_lstm_cell_split_readvariableop_resource/rnn_1_lstm_cell_split_1_readvariableop_resource'rnn_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *%
_read_only_resource_inputs

*#
bodyR
rnn_1_while_body_482103*#
condR
rnn_1_while_cond_482102*`
output_shapesO
M: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 2
rnn_1/whileÁ
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   28
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shape
(rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_1/while:output:3?rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02*
(rnn_1/TensorArrayV2Stack/TensorListStack
rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn_1/strided_slice_3/stack
rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_3/stack_1
rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_3/stack_2¾
rnn_1/strided_slice_3StridedSlice1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_1/strided_slice_3/stack:output:0&rnn_1/strided_slice_3/stack_1:output:0&rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
rnn_1/strided_slice_3
rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose_2/permÆ
rnn_1/transpose_2	Transpose1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
rnn_1/transpose_2
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulrnn_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxy
IdentityIdentitydense/Softmax:softmax:0^rnn_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::2
rnn_1/whilernn_1/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ºÝ

!__inference__wrapped_model_480494
masking_input<
8sequential_rnn_1_lstm_cell_split_readvariableop_resource>
:sequential_rnn_1_lstm_cell_split_1_readvariableop_resource6
2sequential_rnn_1_lstm_cell_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity¢sequential/rnn_1/while
sequential/masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/masking/NotEqual/y½
sequential/masking/NotEqualNotEqualmasking_input&sequential/masking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
sequential/masking/NotEqual
(sequential/masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/masking/Any/reduction_indicesÒ
sequential/masking/AnyAnysequential/masking/NotEqual:z:01sequential/masking/Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
sequential/masking/Any©
sequential/masking/CastCastsequential/masking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/masking/Cast£
sequential/masking/mulMulmasking_inputsequential/masking/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
sequential/masking/mul¿
sequential/masking/SqueezeSqueezesequential/masking/Any:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
sequential/masking/Squeezez
sequential/rnn_1/ShapeShapesequential/masking/mul:z:0*
T0*
_output_shapes
:2
sequential/rnn_1/Shape
$sequential/rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/rnn_1/strided_slice/stack
&sequential/rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn_1/strided_slice/stack_1
&sequential/rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn_1/strided_slice/stack_2È
sequential/rnn_1/strided_sliceStridedSlicesequential/rnn_1/Shape:output:0-sequential/rnn_1/strided_slice/stack:output:0/sequential/rnn_1/strided_slice/stack_1:output:0/sequential/rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/rnn_1/strided_slice~
sequential/rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
sequential/rnn_1/zeros/mul/y°
sequential/rnn_1/zeros/mulMul'sequential/rnn_1/strided_slice:output:0%sequential/rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros/mul
sequential/rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/rnn_1/zeros/Less/y«
sequential/rnn_1/zeros/LessLesssequential/rnn_1/zeros/mul:z:0&sequential/rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros/Less
sequential/rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2!
sequential/rnn_1/zeros/packed/1Ç
sequential/rnn_1/zeros/packedPack'sequential/rnn_1/strided_slice:output:0(sequential/rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/rnn_1/zeros/packed
sequential/rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/rnn_1/zeros/Const¹
sequential/rnn_1/zerosFill&sequential/rnn_1/zeros/packed:output:0%sequential/rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential/rnn_1/zeros
sequential/rnn_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2 
sequential/rnn_1/zeros_1/mul/y¶
sequential/rnn_1/zeros_1/mulMul'sequential/rnn_1/strided_slice:output:0'sequential/rnn_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros_1/mul
sequential/rnn_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2!
sequential/rnn_1/zeros_1/Less/y³
sequential/rnn_1/zeros_1/LessLess sequential/rnn_1/zeros_1/mul:z:0(sequential/rnn_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros_1/Less
!sequential/rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential/rnn_1/zeros_1/packed/1Í
sequential/rnn_1/zeros_1/packedPack'sequential/rnn_1/strided_slice:output:0*sequential/rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
sequential/rnn_1/zeros_1/packed
sequential/rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential/rnn_1/zeros_1/ConstÁ
sequential/rnn_1/zeros_1Fill(sequential/rnn_1/zeros_1/packed:output:0'sequential/rnn_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential/rnn_1/zeros_1
sequential/rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/rnn_1/transpose/permË
sequential/rnn_1/transpose	Transposesequential/masking/mul:z:0(sequential/rnn_1/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
sequential/rnn_1/transpose
sequential/rnn_1/Shape_1Shapesequential/rnn_1/transpose:y:0*
T0*
_output_shapes
:2
sequential/rnn_1/Shape_1
&sequential/rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/rnn_1/strided_slice_1/stack
(sequential/rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_1/stack_1
(sequential/rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_1/stack_2Ô
 sequential/rnn_1/strided_slice_1StridedSlice!sequential/rnn_1/Shape_1:output:0/sequential/rnn_1/strided_slice_1/stack:output:01sequential/rnn_1/strided_slice_1/stack_1:output:01sequential/rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/rnn_1/strided_slice_1
sequential/rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
sequential/rnn_1/ExpandDims/dimÖ
sequential/rnn_1/ExpandDims
ExpandDims#sequential/masking/Squeeze:output:0(sequential/rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/rnn_1/ExpandDims
!sequential/rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential/rnn_1/transpose_1/permÚ
sequential/rnn_1/transpose_1	Transpose$sequential/rnn_1/ExpandDims:output:0*sequential/rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/rnn_1/transpose_1§
,sequential/rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,sequential/rnn_1/TensorArrayV2/element_shapeö
sequential/rnn_1/TensorArrayV2TensorListReserve5sequential/rnn_1/TensorArrayV2/element_shape:output:0)sequential/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/rnn_1/TensorArrayV2á
Fsequential/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2H
Fsequential/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape¼
8sequential/rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/rnn_1/transpose:y:0Osequential/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8sequential/rnn_1/TensorArrayUnstack/TensorListFromTensor
&sequential/rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/rnn_1/strided_slice_2/stack
(sequential/rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_2/stack_1
(sequential/rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_2/stack_2ã
 sequential/rnn_1/strided_slice_2StridedSlicesequential/rnn_1/transpose:y:0/sequential/rnn_1/strided_slice_2/stack:output:01sequential/rnn_1/strided_slice_2/stack_1:output:01sequential/rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2"
 sequential/rnn_1/strided_slice_2§
*sequential/rnn_1/lstm_cell/ones_like/ShapeShapesequential/rnn_1/zeros:output:0*
T0*
_output_shapes
:2,
*sequential/rnn_1/lstm_cell/ones_like/Shape
*sequential/rnn_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*sequential/rnn_1/lstm_cell/ones_like/Constð
$sequential/rnn_1/lstm_cell/ones_likeFill3sequential/rnn_1/lstm_cell/ones_like/Shape:output:03sequential/rnn_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/lstm_cell/ones_like
 sequential/rnn_1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential/rnn_1/lstm_cell/Const
*sequential/rnn_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*sequential/rnn_1/lstm_cell/split/split_dimÝ
/sequential/rnn_1/lstm_cell/split/ReadVariableOpReadVariableOp8sequential_rnn_1_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype021
/sequential/rnn_1/lstm_cell/split/ReadVariableOp
 sequential/rnn_1/lstm_cell/splitSplit3sequential/rnn_1/lstm_cell/split/split_dim:output:07sequential/rnn_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2"
 sequential/rnn_1/lstm_cell/splitØ
!sequential/rnn_1/lstm_cell/MatMulMatMul)sequential/rnn_1/strided_slice_2:output:0)sequential/rnn_1/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/lstm_cell/MatMulÜ
#sequential/rnn_1/lstm_cell/MatMul_1MatMul)sequential/rnn_1/strided_slice_2:output:0)sequential/rnn_1/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_1Ü
#sequential/rnn_1/lstm_cell/MatMul_2MatMul)sequential/rnn_1/strided_slice_2:output:0)sequential/rnn_1/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_2Ü
#sequential/rnn_1/lstm_cell/MatMul_3MatMul)sequential/rnn_1/strided_slice_2:output:0)sequential/rnn_1/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_3
"sequential/rnn_1/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/rnn_1/lstm_cell/Const_1
,sequential/rnn_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/rnn_1/lstm_cell/split_1/split_dimÞ
1sequential/rnn_1/lstm_cell/split_1/ReadVariableOpReadVariableOp:sequential_rnn_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential/rnn_1/lstm_cell/split_1/ReadVariableOp
"sequential/rnn_1/lstm_cell/split_1Split5sequential/rnn_1/lstm_cell/split_1/split_dim:output:09sequential/rnn_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2$
"sequential/rnn_1/lstm_cell/split_1ß
"sequential/rnn_1/lstm_cell/BiasAddBiasAdd+sequential/rnn_1/lstm_cell/MatMul:product:0+sequential/rnn_1/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"sequential/rnn_1/lstm_cell/BiasAddå
$sequential/rnn_1/lstm_cell/BiasAdd_1BiasAdd-sequential/rnn_1/lstm_cell/MatMul_1:product:0+sequential/rnn_1/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/lstm_cell/BiasAdd_1å
$sequential/rnn_1/lstm_cell/BiasAdd_2BiasAdd-sequential/rnn_1/lstm_cell/MatMul_2:product:0+sequential/rnn_1/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/lstm_cell/BiasAdd_2å
$sequential/rnn_1/lstm_cell/BiasAdd_3BiasAdd-sequential/rnn_1/lstm_cell/MatMul_3:product:0+sequential/rnn_1/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/lstm_cell/BiasAdd_3É
sequential/rnn_1/lstm_cell/mulMulsequential/rnn_1/zeros:output:0-sequential/rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential/rnn_1/lstm_cell/mulÍ
 sequential/rnn_1/lstm_cell/mul_1Mulsequential/rnn_1/zeros:output:0-sequential/rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/mul_1Í
 sequential/rnn_1/lstm_cell/mul_2Mulsequential/rnn_1/zeros:output:0-sequential/rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/mul_2Í
 sequential/rnn_1/lstm_cell/mul_3Mulsequential/rnn_1/zeros:output:0-sequential/rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/mul_3Ê
)sequential/rnn_1/lstm_cell/ReadVariableOpReadVariableOp2sequential_rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02+
)sequential/rnn_1/lstm_cell/ReadVariableOp±
.sequential/rnn_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.sequential/rnn_1/lstm_cell/strided_slice/stackµ
0sequential/rnn_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   22
0sequential/rnn_1/lstm_cell/strided_slice/stack_1µ
0sequential/rnn_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0sequential/rnn_1/lstm_cell/strided_slice/stack_2
(sequential/rnn_1/lstm_cell/strided_sliceStridedSlice1sequential/rnn_1/lstm_cell/ReadVariableOp:value:07sequential/rnn_1/lstm_cell/strided_slice/stack:output:09sequential/rnn_1/lstm_cell/strided_slice/stack_1:output:09sequential/rnn_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2*
(sequential/rnn_1/lstm_cell/strided_sliceÝ
#sequential/rnn_1/lstm_cell/MatMul_4MatMul"sequential/rnn_1/lstm_cell/mul:z:01sequential/rnn_1/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_4×
sequential/rnn_1/lstm_cell/addAddV2+sequential/rnn_1/lstm_cell/BiasAdd:output:0-sequential/rnn_1/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential/rnn_1/lstm_cell/add©
"sequential/rnn_1/lstm_cell/SigmoidSigmoid"sequential/rnn_1/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"sequential/rnn_1/lstm_cell/SigmoidÎ
+sequential/rnn_1/lstm_cell/ReadVariableOp_1ReadVariableOp2sequential_rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02-
+sequential/rnn_1/lstm_cell/ReadVariableOp_1µ
0sequential/rnn_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0sequential/rnn_1/lstm_cell/strided_slice_1/stack¹
2sequential/rnn_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/rnn_1/lstm_cell/strided_slice_1/stack_1¹
2sequential/rnn_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2sequential/rnn_1/lstm_cell/strided_slice_1/stack_2ª
*sequential/rnn_1/lstm_cell/strided_slice_1StridedSlice3sequential/rnn_1/lstm_cell/ReadVariableOp_1:value:09sequential/rnn_1/lstm_cell/strided_slice_1/stack:output:0;sequential/rnn_1/lstm_cell/strided_slice_1/stack_1:output:0;sequential/rnn_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2,
*sequential/rnn_1/lstm_cell/strided_slice_1á
#sequential/rnn_1/lstm_cell/MatMul_5MatMul$sequential/rnn_1/lstm_cell/mul_1:z:03sequential/rnn_1/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_5Ý
 sequential/rnn_1/lstm_cell/add_1AddV2-sequential/rnn_1/lstm_cell/BiasAdd_1:output:0-sequential/rnn_1/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/add_1¯
$sequential/rnn_1/lstm_cell/Sigmoid_1Sigmoid$sequential/rnn_1/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/lstm_cell/Sigmoid_1Ê
 sequential/rnn_1/lstm_cell/mul_4Mul(sequential/rnn_1/lstm_cell/Sigmoid_1:y:0!sequential/rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/mul_4Î
+sequential/rnn_1/lstm_cell/ReadVariableOp_2ReadVariableOp2sequential_rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02-
+sequential/rnn_1/lstm_cell/ReadVariableOp_2µ
0sequential/rnn_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0sequential/rnn_1/lstm_cell/strided_slice_2/stack¹
2sequential/rnn_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   24
2sequential/rnn_1/lstm_cell/strided_slice_2/stack_1¹
2sequential/rnn_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2sequential/rnn_1/lstm_cell/strided_slice_2/stack_2ª
*sequential/rnn_1/lstm_cell/strided_slice_2StridedSlice3sequential/rnn_1/lstm_cell/ReadVariableOp_2:value:09sequential/rnn_1/lstm_cell/strided_slice_2/stack:output:0;sequential/rnn_1/lstm_cell/strided_slice_2/stack_1:output:0;sequential/rnn_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2,
*sequential/rnn_1/lstm_cell/strided_slice_2á
#sequential/rnn_1/lstm_cell/MatMul_6MatMul$sequential/rnn_1/lstm_cell/mul_2:z:03sequential/rnn_1/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_6Ý
 sequential/rnn_1/lstm_cell/add_2AddV2-sequential/rnn_1/lstm_cell/BiasAdd_2:output:0-sequential/rnn_1/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/add_2¢
sequential/rnn_1/lstm_cell/TanhTanh$sequential/rnn_1/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential/rnn_1/lstm_cell/TanhÊ
 sequential/rnn_1/lstm_cell/mul_5Mul&sequential/rnn_1/lstm_cell/Sigmoid:y:0#sequential/rnn_1/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/mul_5Ë
 sequential/rnn_1/lstm_cell/add_3AddV2$sequential/rnn_1/lstm_cell/mul_4:z:0$sequential/rnn_1/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/add_3Î
+sequential/rnn_1/lstm_cell/ReadVariableOp_3ReadVariableOp2sequential_rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02-
+sequential/rnn_1/lstm_cell/ReadVariableOp_3µ
0sequential/rnn_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   22
0sequential/rnn_1/lstm_cell/strided_slice_3/stack¹
2sequential/rnn_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2sequential/rnn_1/lstm_cell/strided_slice_3/stack_1¹
2sequential/rnn_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2sequential/rnn_1/lstm_cell/strided_slice_3/stack_2ª
*sequential/rnn_1/lstm_cell/strided_slice_3StridedSlice3sequential/rnn_1/lstm_cell/ReadVariableOp_3:value:09sequential/rnn_1/lstm_cell/strided_slice_3/stack:output:0;sequential/rnn_1/lstm_cell/strided_slice_3/stack_1:output:0;sequential/rnn_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2,
*sequential/rnn_1/lstm_cell/strided_slice_3á
#sequential/rnn_1/lstm_cell/MatMul_7MatMul$sequential/rnn_1/lstm_cell/mul_3:z:03sequential/rnn_1/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#sequential/rnn_1/lstm_cell/MatMul_7Ý
 sequential/rnn_1/lstm_cell/add_4AddV2-sequential/rnn_1/lstm_cell/BiasAdd_3:output:0-sequential/rnn_1/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/add_4¯
$sequential/rnn_1/lstm_cell/Sigmoid_2Sigmoid$sequential/rnn_1/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/lstm_cell/Sigmoid_2¦
!sequential/rnn_1/lstm_cell/Tanh_1Tanh$sequential/rnn_1/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/lstm_cell/Tanh_1Î
 sequential/rnn_1/lstm_cell/mul_6Mul(sequential/rnn_1/lstm_cell/Sigmoid_2:y:0%sequential/rnn_1/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 sequential/rnn_1/lstm_cell/mul_6±
.sequential/rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   20
.sequential/rnn_1/TensorArrayV2_1/element_shapeü
 sequential/rnn_1/TensorArrayV2_1TensorListReserve7sequential/rnn_1/TensorArrayV2_1/element_shape:output:0)sequential/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sequential/rnn_1/TensorArrayV2_1p
sequential/rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn_1/time«
.sequential/rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/rnn_1/TensorArrayV2_2/element_shapeü
 sequential/rnn_1/TensorArrayV2_2TensorListReserve7sequential/rnn_1/TensorArrayV2_2/element_shape:output:0)sequential/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02"
 sequential/rnn_1/TensorArrayV2_2å
Hsequential/rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2J
Hsequential/rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeÄ
:sequential/rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor sequential/rnn_1/transpose_1:y:0Qsequential/rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02<
:sequential/rnn_1/TensorArrayUnstack_1/TensorListFromTensor
sequential/rnn_1/zeros_like	ZerosLike$sequential/rnn_1/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential/rnn_1/zeros_like¡
)sequential/rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)sequential/rnn_1/while/maximum_iterations
#sequential/rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/rnn_1/while/loop_counteró
sequential/rnn_1/whileWhile,sequential/rnn_1/while/loop_counter:output:02sequential/rnn_1/while/maximum_iterations:output:0sequential/rnn_1/time:output:0)sequential/rnn_1/TensorArrayV2_1:handle:0sequential/rnn_1/zeros_like:y:0sequential/rnn_1/zeros:output:0!sequential/rnn_1/zeros_1:output:0)sequential/rnn_1/strided_slice_1:output:0Hsequential/rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential/rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:08sequential_rnn_1_lstm_cell_split_readvariableop_resource:sequential_rnn_1_lstm_cell_split_1_readvariableop_resource2sequential_rnn_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *%
_read_only_resource_inputs

*.
body&R$
"sequential_rnn_1_while_body_480341*.
cond&R$
"sequential_rnn_1_while_cond_480340*`
output_shapesO
M: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 2
sequential/rnn_1/while×
Asequential/rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2C
Asequential/rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeµ
3sequential/rnn_1/TensorArrayV2Stack/TensorListStackTensorListStacksequential/rnn_1/while:output:3Jsequential/rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype025
3sequential/rnn_1/TensorArrayV2Stack/TensorListStack£
&sequential/rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2(
&sequential/rnn_1/strided_slice_3/stack
(sequential/rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/rnn_1/strided_slice_3/stack_1
(sequential/rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_3/stack_2
 sequential/rnn_1/strided_slice_3StridedSlice<sequential/rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/rnn_1/strided_slice_3/stack:output:01sequential/rnn_1/strided_slice_3/stack_1:output:01sequential/rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2"
 sequential/rnn_1/strided_slice_3
!sequential/rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential/rnn_1/transpose_2/permò
sequential/rnn_1/transpose_2	Transpose<sequential/rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0*sequential/rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
sequential/rnn_1/transpose_2À
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&sequential/dense/MatMul/ReadVariableOpÉ
sequential/dense/MatMulMatMul)sequential/rnn_1/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMul¿
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Softmax
IdentityIdentity"sequential/dense/Softmax:softmax:0^sequential/rnn_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::20
sequential/rnn_1/whilesequential/rnn_1/while:d `
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'
_user_specified_namemasking_input
È
D
(__inference_masking_layer_call_fn_482615

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_masking_layer_call_and_return_conditional_losses_4812352
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ú·

"sequential_rnn_1_while_body_480341>
:sequential_rnn_1_while_sequential_rnn_1_while_loop_counterD
@sequential_rnn_1_while_sequential_rnn_1_while_maximum_iterations&
"sequential_rnn_1_while_placeholder(
$sequential_rnn_1_while_placeholder_1(
$sequential_rnn_1_while_placeholder_2(
$sequential_rnn_1_while_placeholder_3(
$sequential_rnn_1_while_placeholder_4=
9sequential_rnn_1_while_sequential_rnn_1_strided_slice_1_0y
usequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor_0}
ysequential_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0D
@sequential_rnn_1_while_lstm_cell_split_readvariableop_resource_0F
Bsequential_rnn_1_while_lstm_cell_split_1_readvariableop_resource_0>
:sequential_rnn_1_while_lstm_cell_readvariableop_resource_0#
sequential_rnn_1_while_identity%
!sequential_rnn_1_while_identity_1%
!sequential_rnn_1_while_identity_2%
!sequential_rnn_1_while_identity_3%
!sequential_rnn_1_while_identity_4%
!sequential_rnn_1_while_identity_5%
!sequential_rnn_1_while_identity_6;
7sequential_rnn_1_while_sequential_rnn_1_strided_slice_1w
ssequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor{
wsequential_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_1_tensorlistfromtensorB
>sequential_rnn_1_while_lstm_cell_split_readvariableop_resourceD
@sequential_rnn_1_while_lstm_cell_split_1_readvariableop_resource<
8sequential_rnn_1_while_lstm_cell_readvariableop_resourceå
Hsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2J
Hsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeº
:sequential/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor_0"sequential_rnn_1_while_placeholderQsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02<
:sequential/rnn_1/while/TensorArrayV2Read/TensorListGetItemé
Jsequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2L
Jsequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeÃ
<sequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemysequential_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"sequential_rnn_1_while_placeholderSsequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2>
<sequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem¸
0sequential/rnn_1/while/lstm_cell/ones_like/ShapeShape$sequential_rnn_1_while_placeholder_3*
T0*
_output_shapes
:22
0sequential/rnn_1/while/lstm_cell/ones_like/Shape©
0sequential/rnn_1/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential/rnn_1/while/lstm_cell/ones_like/Const
*sequential/rnn_1/while/lstm_cell/ones_likeFill9sequential/rnn_1/while/lstm_cell/ones_like/Shape:output:09sequential/rnn_1/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*sequential/rnn_1/while/lstm_cell/ones_like
&sequential/rnn_1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential/rnn_1/while/lstm_cell/Const¦
0sequential/rnn_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/rnn_1/while/lstm_cell/split/split_dimñ
5sequential/rnn_1/while/lstm_cell/split/ReadVariableOpReadVariableOp@sequential_rnn_1_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype027
5sequential/rnn_1/while/lstm_cell/split/ReadVariableOp¯
&sequential/rnn_1/while/lstm_cell/splitSplit9sequential/rnn_1/while/lstm_cell/split/split_dim:output:0=sequential/rnn_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2(
&sequential/rnn_1/while/lstm_cell/split
'sequential/rnn_1/while/lstm_cell/MatMulMatMulAsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential/rnn_1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential/rnn_1/while/lstm_cell/MatMul
)sequential/rnn_1/while/lstm_cell/MatMul_1MatMulAsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential/rnn_1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_1
)sequential/rnn_1/while/lstm_cell/MatMul_2MatMulAsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential/rnn_1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_2
)sequential/rnn_1/while/lstm_cell/MatMul_3MatMulAsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential/rnn_1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_3
(sequential/rnn_1/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential/rnn_1/while/lstm_cell/Const_1ª
2sequential/rnn_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential/rnn_1/while/lstm_cell/split_1/split_dimò
7sequential/rnn_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOpBsequential_rnn_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype029
7sequential/rnn_1/while/lstm_cell/split_1/ReadVariableOp£
(sequential/rnn_1/while/lstm_cell/split_1Split;sequential/rnn_1/while/lstm_cell/split_1/split_dim:output:0?sequential/rnn_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2*
(sequential/rnn_1/while/lstm_cell/split_1÷
(sequential/rnn_1/while/lstm_cell/BiasAddBiasAdd1sequential/rnn_1/while/lstm_cell/MatMul:product:01sequential/rnn_1/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(sequential/rnn_1/while/lstm_cell/BiasAddý
*sequential/rnn_1/while/lstm_cell/BiasAdd_1BiasAdd3sequential/rnn_1/while/lstm_cell/MatMul_1:product:01sequential/rnn_1/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*sequential/rnn_1/while/lstm_cell/BiasAdd_1ý
*sequential/rnn_1/while/lstm_cell/BiasAdd_2BiasAdd3sequential/rnn_1/while/lstm_cell/MatMul_2:product:01sequential/rnn_1/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*sequential/rnn_1/while/lstm_cell/BiasAdd_2ý
*sequential/rnn_1/while/lstm_cell/BiasAdd_3BiasAdd3sequential/rnn_1/while/lstm_cell/MatMul_3:product:01sequential/rnn_1/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*sequential/rnn_1/while/lstm_cell/BiasAdd_3à
$sequential/rnn_1/while/lstm_cell/mulMul$sequential_rnn_1_while_placeholder_33sequential/rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/while/lstm_cell/mulä
&sequential/rnn_1/while/lstm_cell/mul_1Mul$sequential_rnn_1_while_placeholder_33sequential/rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/mul_1ä
&sequential/rnn_1/while/lstm_cell/mul_2Mul$sequential_rnn_1_while_placeholder_33sequential/rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/mul_2ä
&sequential/rnn_1/while/lstm_cell/mul_3Mul$sequential_rnn_1_while_placeholder_33sequential/rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/mul_3Þ
/sequential/rnn_1/while/lstm_cell/ReadVariableOpReadVariableOp:sequential_rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype021
/sequential/rnn_1/while/lstm_cell/ReadVariableOp½
4sequential/rnn_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4sequential/rnn_1/while/lstm_cell/strided_slice/stackÁ
6sequential/rnn_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   28
6sequential/rnn_1/while/lstm_cell/strided_slice/stack_1Á
6sequential/rnn_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential/rnn_1/while/lstm_cell/strided_slice/stack_2Â
.sequential/rnn_1/while/lstm_cell/strided_sliceStridedSlice7sequential/rnn_1/while/lstm_cell/ReadVariableOp:value:0=sequential/rnn_1/while/lstm_cell/strided_slice/stack:output:0?sequential/rnn_1/while/lstm_cell/strided_slice/stack_1:output:0?sequential/rnn_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask20
.sequential/rnn_1/while/lstm_cell/strided_sliceõ
)sequential/rnn_1/while/lstm_cell/MatMul_4MatMul(sequential/rnn_1/while/lstm_cell/mul:z:07sequential/rnn_1/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_4ï
$sequential/rnn_1/while/lstm_cell/addAddV21sequential/rnn_1/while/lstm_cell/BiasAdd:output:03sequential/rnn_1/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential/rnn_1/while/lstm_cell/add»
(sequential/rnn_1/while/lstm_cell/SigmoidSigmoid(sequential/rnn_1/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(sequential/rnn_1/while/lstm_cell/Sigmoidâ
1sequential/rnn_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp:sequential_rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype023
1sequential/rnn_1/while/lstm_cell/ReadVariableOp_1Á
6sequential/rnn_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   28
6sequential/rnn_1/while/lstm_cell/strided_slice_1/stackÅ
8sequential/rnn_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2:
8sequential/rnn_1/while/lstm_cell/strided_slice_1/stack_1Å
8sequential/rnn_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/rnn_1/while/lstm_cell/strided_slice_1/stack_2Î
0sequential/rnn_1/while/lstm_cell/strided_slice_1StridedSlice9sequential/rnn_1/while/lstm_cell/ReadVariableOp_1:value:0?sequential/rnn_1/while/lstm_cell/strided_slice_1/stack:output:0Asequential/rnn_1/while/lstm_cell/strided_slice_1/stack_1:output:0Asequential/rnn_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask22
0sequential/rnn_1/while/lstm_cell/strided_slice_1ù
)sequential/rnn_1/while/lstm_cell/MatMul_5MatMul*sequential/rnn_1/while/lstm_cell/mul_1:z:09sequential/rnn_1/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_5õ
&sequential/rnn_1/while/lstm_cell/add_1AddV23sequential/rnn_1/while/lstm_cell/BiasAdd_1:output:03sequential/rnn_1/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/add_1Á
*sequential/rnn_1/while/lstm_cell/Sigmoid_1Sigmoid*sequential/rnn_1/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*sequential/rnn_1/while/lstm_cell/Sigmoid_1ß
&sequential/rnn_1/while/lstm_cell/mul_4Mul.sequential/rnn_1/while/lstm_cell/Sigmoid_1:y:0$sequential_rnn_1_while_placeholder_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/mul_4â
1sequential/rnn_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp:sequential_rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype023
1sequential/rnn_1/while/lstm_cell/ReadVariableOp_2Á
6sequential/rnn_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential/rnn_1/while/lstm_cell/strided_slice_2/stackÅ
8sequential/rnn_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2:
8sequential/rnn_1/while/lstm_cell/strided_slice_2/stack_1Å
8sequential/rnn_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/rnn_1/while/lstm_cell/strided_slice_2/stack_2Î
0sequential/rnn_1/while/lstm_cell/strided_slice_2StridedSlice9sequential/rnn_1/while/lstm_cell/ReadVariableOp_2:value:0?sequential/rnn_1/while/lstm_cell/strided_slice_2/stack:output:0Asequential/rnn_1/while/lstm_cell/strided_slice_2/stack_1:output:0Asequential/rnn_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask22
0sequential/rnn_1/while/lstm_cell/strided_slice_2ù
)sequential/rnn_1/while/lstm_cell/MatMul_6MatMul*sequential/rnn_1/while/lstm_cell/mul_2:z:09sequential/rnn_1/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_6õ
&sequential/rnn_1/while/lstm_cell/add_2AddV23sequential/rnn_1/while/lstm_cell/BiasAdd_2:output:03sequential/rnn_1/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/add_2´
%sequential/rnn_1/while/lstm_cell/TanhTanh*sequential/rnn_1/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%sequential/rnn_1/while/lstm_cell/Tanhâ
&sequential/rnn_1/while/lstm_cell/mul_5Mul,sequential/rnn_1/while/lstm_cell/Sigmoid:y:0)sequential/rnn_1/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/mul_5ã
&sequential/rnn_1/while/lstm_cell/add_3AddV2*sequential/rnn_1/while/lstm_cell/mul_4:z:0*sequential/rnn_1/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/add_3â
1sequential/rnn_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp:sequential_rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype023
1sequential/rnn_1/while/lstm_cell/ReadVariableOp_3Á
6sequential/rnn_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   28
6sequential/rnn_1/while/lstm_cell/strided_slice_3/stackÅ
8sequential/rnn_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential/rnn_1/while/lstm_cell/strided_slice_3/stack_1Å
8sequential/rnn_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/rnn_1/while/lstm_cell/strided_slice_3/stack_2Î
0sequential/rnn_1/while/lstm_cell/strided_slice_3StridedSlice9sequential/rnn_1/while/lstm_cell/ReadVariableOp_3:value:0?sequential/rnn_1/while/lstm_cell/strided_slice_3/stack:output:0Asequential/rnn_1/while/lstm_cell/strided_slice_3/stack_1:output:0Asequential/rnn_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask22
0sequential/rnn_1/while/lstm_cell/strided_slice_3ù
)sequential/rnn_1/while/lstm_cell/MatMul_7MatMul*sequential/rnn_1/while/lstm_cell/mul_3:z:09sequential/rnn_1/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)sequential/rnn_1/while/lstm_cell/MatMul_7õ
&sequential/rnn_1/while/lstm_cell/add_4AddV23sequential/rnn_1/while/lstm_cell/BiasAdd_3:output:03sequential/rnn_1/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/add_4Á
*sequential/rnn_1/while/lstm_cell/Sigmoid_2Sigmoid*sequential/rnn_1/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*sequential/rnn_1/while/lstm_cell/Sigmoid_2¸
'sequential/rnn_1/while/lstm_cell/Tanh_1Tanh*sequential/rnn_1/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential/rnn_1/while/lstm_cell/Tanh_1æ
&sequential/rnn_1/while/lstm_cell/mul_6Mul.sequential/rnn_1/while/lstm_cell/Sigmoid_2:y:0+sequential/rnn_1/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&sequential/rnn_1/while/lstm_cell/mul_6
%sequential/rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2'
%sequential/rnn_1/while/Tile/multiplesé
sequential/rnn_1/while/TileTileCsequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0.sequential/rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rnn_1/while/Tileø
sequential/rnn_1/while/SelectV2SelectV2$sequential/rnn_1/while/Tile:output:0*sequential/rnn_1/while/lstm_cell/mul_6:z:0$sequential_rnn_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential/rnn_1/while/SelectV2£
'sequential/rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/rnn_1/while/Tile_1/multiplesï
sequential/rnn_1/while/Tile_1TileCsequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:00sequential/rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rnn_1/while/Tile_1£
'sequential/rnn_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/rnn_1/while/Tile_2/multiplesï
sequential/rnn_1/while/Tile_2TileCsequential/rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:00sequential/rnn_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rnn_1/while/Tile_2þ
!sequential/rnn_1/while/SelectV2_1SelectV2&sequential/rnn_1/while/Tile_1:output:0*sequential/rnn_1/while/lstm_cell/mul_6:z:0$sequential_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/while/SelectV2_1þ
!sequential/rnn_1/while/SelectV2_2SelectV2&sequential/rnn_1/while/Tile_2:output:0*sequential/rnn_1/while/lstm_cell/add_3:z:0$sequential_rnn_1_while_placeholder_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/while/SelectV2_2°
;sequential/rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sequential_rnn_1_while_placeholder_1"sequential_rnn_1_while_placeholder(sequential/rnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype02=
;sequential/rnn_1/while/TensorArrayV2Write/TensorListSetItem~
sequential/rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/rnn_1/while/add/y­
sequential/rnn_1/while/addAddV2"sequential_rnn_1_while_placeholder%sequential/rnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/while/add
sequential/rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential/rnn_1/while/add_1/yË
sequential/rnn_1/while/add_1AddV2:sequential_rnn_1_while_sequential_rnn_1_while_loop_counter'sequential/rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/while/add_1
sequential/rnn_1/while/IdentityIdentity sequential/rnn_1/while/add_1:z:0*
T0*
_output_shapes
: 2!
sequential/rnn_1/while/Identityµ
!sequential/rnn_1/while/Identity_1Identity@sequential_rnn_1_while_sequential_rnn_1_while_maximum_iterations*
T0*
_output_shapes
: 2#
!sequential/rnn_1/while/Identity_1
!sequential/rnn_1/while/Identity_2Identitysequential/rnn_1/while/add:z:0*
T0*
_output_shapes
: 2#
!sequential/rnn_1/while/Identity_2À
!sequential/rnn_1/while/Identity_3IdentityKsequential/rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2#
!sequential/rnn_1/while/Identity_3®
!sequential/rnn_1/while/Identity_4Identity(sequential/rnn_1/while/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/while/Identity_4°
!sequential/rnn_1/while/Identity_5Identity*sequential/rnn_1/while/SelectV2_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/while/Identity_5°
!sequential/rnn_1/while/Identity_6Identity*sequential/rnn_1/while/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential/rnn_1/while/Identity_6"K
sequential_rnn_1_while_identity(sequential/rnn_1/while/Identity:output:0"O
!sequential_rnn_1_while_identity_1*sequential/rnn_1/while/Identity_1:output:0"O
!sequential_rnn_1_while_identity_2*sequential/rnn_1/while/Identity_2:output:0"O
!sequential_rnn_1_while_identity_3*sequential/rnn_1/while/Identity_3:output:0"O
!sequential_rnn_1_while_identity_4*sequential/rnn_1/while/Identity_4:output:0"O
!sequential_rnn_1_while_identity_5*sequential/rnn_1/while/Identity_5:output:0"O
!sequential_rnn_1_while_identity_6*sequential/rnn_1/while/Identity_6:output:0"v
8sequential_rnn_1_while_lstm_cell_readvariableop_resource:sequential_rnn_1_while_lstm_cell_readvariableop_resource_0"
@sequential_rnn_1_while_lstm_cell_split_1_readvariableop_resourceBsequential_rnn_1_while_lstm_cell_split_1_readvariableop_resource_0"
>sequential_rnn_1_while_lstm_cell_split_readvariableop_resource@sequential_rnn_1_while_lstm_cell_split_readvariableop_resource_0"t
7sequential_rnn_1_while_sequential_rnn_1_strided_slice_19sequential_rnn_1_while_sequential_rnn_1_strided_slice_1_0"ô
wsequential_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_1_tensorlistfromtensorysequential_rnn_1_while_tensorarrayv2read_1_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"ì
ssequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensorusequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
¶
í
A__inference_rnn_1_layer_call_and_return_conditional_losses_482919
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const¬
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout/Const§
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeñ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ì¯20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 lstm_cell/dropout/GreaterEqual/yæ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Cast¢
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeö
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ÜË[22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_1/GreaterEqual/yî
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_1/GreaterEqual£
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Castª
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shape÷
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Ñô22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_2/GreaterEqual/yî
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_2/GreaterEqual£
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Castª
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape÷
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Àï22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2$
"lstm_cell/dropout_3/GreaterEqual/yî
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 lstm_cell/dropout_3/GreaterEqual£
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Castª
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/dropout_3/Mul_1d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02 
lstm_cell/split/ReadVariableOpÓ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpÇ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd¡
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_1¡
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_2¡
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_3
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_1
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_2
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2¸
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_1
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_4
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_2o
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_5
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_2s
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh_1
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_482759*
condR
while_cond_482758*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1t
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
§

rnn_1_while_cond_482415(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3
rnn_1_while_placeholder_4*
&rnn_1_while_less_rnn_1_strided_slice_1@
<rnn_1_while_rnn_1_while_cond_482415___redundant_placeholder0@
<rnn_1_while_rnn_1_while_cond_482415___redundant_placeholder1@
<rnn_1_while_rnn_1_while_cond_482415___redundant_placeholder2@
<rnn_1_while_rnn_1_while_cond_482415___redundant_placeholder3@
<rnn_1_while_rnn_1_while_cond_482415___redundant_placeholder4
rnn_1_while_identity

rnn_1/while/LessLessrnn_1_while_placeholder&rnn_1_while_less_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
rnn_1/while/Lesso
rnn_1/while/IdentityIdentityrnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn_1/while/Identity"5
rnn_1_while_identityrnn_1/while/Identity:output:0*j
_input_shapesY
W: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ý£
Ô
while_body_482759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstÄ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/ones_like
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/dropout/Const¿
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Mul
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2¡26
4while/lstm_cell/dropout/random_uniform/RandomUniform
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2(
&while/lstm_cell/dropout/GreaterEqual/yþ
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$while/lstm_cell/dropout/GreaterEqual¯
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Castº
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout/Mul_1
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_1/ConstÅ
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_1/Mul
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2°éÐ28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_1/GreaterEqual/y
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_1/GreaterEqualµ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_1/CastÂ
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_1/Mul_1
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_2/ConstÅ
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_2/Mul
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2¬õ28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_2/GreaterEqual/y
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_2/GreaterEqualµ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_2/CastÂ
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_2/Mul_1
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_3/ConstÅ
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/dropout_3/Mul
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Û28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(while/lstm_cell/dropout_3/GreaterEqual/y
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&while/lstm_cell/dropout_3/GreaterEqualµ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
while/lstm_cell/dropout_3/CastÂ
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
while/lstm_cell/dropout_3/Mul_1p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02&
$while/lstm_cell/split/ReadVariableOpë
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
while/lstm_cell/split¾
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÂ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1Â
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_2Â
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpß
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell/split_1³
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd¹
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_1¹
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_2¹
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_3
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul¡
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_1¡
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_2¡
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_3«
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Ü
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_4«
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid¯
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2è
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_4¯
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2è
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_5
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_3¯
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2è
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh_1¢
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_6Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
®
©
A__inference_dense_layer_call_and_return_conditional_losses_483758

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«
Ã
while_cond_481385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481385___redundant_placeholder04
0while_while_cond_481385___redundant_placeholder14
0while_while_cond_481385___redundant_placeholder24
0while_while_cond_481385___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:



rnn_1_while_body_482416(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3
rnn_1_while_placeholder_4'
#rnn_1_while_rnn_1_strided_slice_1_0c
_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0g
crnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_09
5rnn_1_while_lstm_cell_split_readvariableop_resource_0;
7rnn_1_while_lstm_cell_split_1_readvariableop_resource_03
/rnn_1_while_lstm_cell_readvariableop_resource_0
rnn_1_while_identity
rnn_1_while_identity_1
rnn_1_while_identity_2
rnn_1_while_identity_3
rnn_1_while_identity_4
rnn_1_while_identity_5
rnn_1_while_identity_6%
!rnn_1_while_rnn_1_strided_slice_1a
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensore
arnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor7
3rnn_1_while_lstm_cell_split_readvariableop_resource9
5rnn_1_while_lstm_cell_split_1_readvariableop_resource1
-rnn_1_while_lstm_cell_readvariableop_resourceÏ
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2?
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeø
/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0rnn_1_while_placeholderFrnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype021
/rnn_1/while/TensorArrayV2Read/TensorListGetItemÓ
?rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?rnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape
1rnn_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemcrnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0rnn_1_while_placeholderHrnn_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
23
1rnn_1/while/TensorArrayV2Read_1/TensorListGetItem
%rnn_1/while/lstm_cell/ones_like/ShapeShapernn_1_while_placeholder_3*
T0*
_output_shapes
:2'
%rnn_1/while/lstm_cell/ones_like/Shape
%rnn_1/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%rnn_1/while/lstm_cell/ones_like/ConstÜ
rnn_1/while/lstm_cell/ones_likeFill.rnn_1/while/lstm_cell/ones_like/Shape:output:0.rnn_1/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/ones_like|
rnn_1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/lstm_cell/Const
%rnn_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%rnn_1/while/lstm_cell/split/split_dimÐ
*rnn_1/while/lstm_cell/split/ReadVariableOpReadVariableOp5rnn_1_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02,
*rnn_1/while/lstm_cell/split/ReadVariableOp
rnn_1/while/lstm_cell/splitSplit.rnn_1/while/lstm_cell/split/split_dim:output:02rnn_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
rnn_1/while/lstm_cell/splitÖ
rnn_1/while/lstm_cell/MatMulMatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/MatMulÚ
rnn_1/while/lstm_cell/MatMul_1MatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_1Ú
rnn_1/while/lstm_cell/MatMul_2MatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_2Ú
rnn_1/while/lstm_cell/MatMul_3MatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0$rnn_1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_3
rnn_1/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/lstm_cell/Const_1
'rnn_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'rnn_1/while/lstm_cell/split_1/split_dimÑ
,rnn_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp7rnn_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02.
,rnn_1/while/lstm_cell/split_1/ReadVariableOp÷
rnn_1/while/lstm_cell/split_1Split0rnn_1/while/lstm_cell/split_1/split_dim:output:04rnn_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
rnn_1/while/lstm_cell/split_1Ë
rnn_1/while/lstm_cell/BiasAddBiasAdd&rnn_1/while/lstm_cell/MatMul:product:0&rnn_1/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/BiasAddÑ
rnn_1/while/lstm_cell/BiasAdd_1BiasAdd(rnn_1/while/lstm_cell/MatMul_1:product:0&rnn_1/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/BiasAdd_1Ñ
rnn_1/while/lstm_cell/BiasAdd_2BiasAdd(rnn_1/while/lstm_cell/MatMul_2:product:0&rnn_1/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/BiasAdd_2Ñ
rnn_1/while/lstm_cell/BiasAdd_3BiasAdd(rnn_1/while/lstm_cell/MatMul_3:product:0&rnn_1/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/BiasAdd_3´
rnn_1/while/lstm_cell/mulMulrnn_1_while_placeholder_3(rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul¸
rnn_1/while/lstm_cell/mul_1Mulrnn_1_while_placeholder_3(rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_1¸
rnn_1/while/lstm_cell/mul_2Mulrnn_1_while_placeholder_3(rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_2¸
rnn_1/while/lstm_cell/mul_3Mulrnn_1_while_placeholder_3(rnn_1/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_3½
$rnn_1/while/lstm_cell/ReadVariableOpReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02&
$rnn_1/while/lstm_cell/ReadVariableOp§
)rnn_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)rnn_1/while/lstm_cell/strided_slice/stack«
+rnn_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2-
+rnn_1/while/lstm_cell/strided_slice/stack_1«
+rnn_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+rnn_1/while/lstm_cell/strided_slice/stack_2
#rnn_1/while/lstm_cell/strided_sliceStridedSlice,rnn_1/while/lstm_cell/ReadVariableOp:value:02rnn_1/while/lstm_cell/strided_slice/stack:output:04rnn_1/while/lstm_cell/strided_slice/stack_1:output:04rnn_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2%
#rnn_1/while/lstm_cell/strided_sliceÉ
rnn_1/while/lstm_cell/MatMul_4MatMulrnn_1/while/lstm_cell/mul:z:0,rnn_1/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_4Ã
rnn_1/while/lstm_cell/addAddV2&rnn_1/while/lstm_cell/BiasAdd:output:0(rnn_1/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add
rnn_1/while/lstm_cell/SigmoidSigmoidrnn_1/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/SigmoidÁ
&rnn_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02(
&rnn_1/while/lstm_cell/ReadVariableOp_1«
+rnn_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2-
+rnn_1/while/lstm_cell/strided_slice_1/stack¯
-rnn_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-rnn_1/while/lstm_cell/strided_slice_1/stack_1¯
-rnn_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-rnn_1/while/lstm_cell/strided_slice_1/stack_2
%rnn_1/while/lstm_cell/strided_slice_1StridedSlice.rnn_1/while/lstm_cell/ReadVariableOp_1:value:04rnn_1/while/lstm_cell/strided_slice_1/stack:output:06rnn_1/while/lstm_cell/strided_slice_1/stack_1:output:06rnn_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2'
%rnn_1/while/lstm_cell/strided_slice_1Í
rnn_1/while/lstm_cell/MatMul_5MatMulrnn_1/while/lstm_cell/mul_1:z:0.rnn_1/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_5É
rnn_1/while/lstm_cell/add_1AddV2(rnn_1/while/lstm_cell/BiasAdd_1:output:0(rnn_1/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_1 
rnn_1/while/lstm_cell/Sigmoid_1Sigmoidrnn_1/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/Sigmoid_1³
rnn_1/while/lstm_cell/mul_4Mul#rnn_1/while/lstm_cell/Sigmoid_1:y:0rnn_1_while_placeholder_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_4Á
&rnn_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02(
&rnn_1/while/lstm_cell/ReadVariableOp_2«
+rnn_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2-
+rnn_1/while/lstm_cell/strided_slice_2/stack¯
-rnn_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2/
-rnn_1/while/lstm_cell/strided_slice_2/stack_1¯
-rnn_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-rnn_1/while/lstm_cell/strided_slice_2/stack_2
%rnn_1/while/lstm_cell/strided_slice_2StridedSlice.rnn_1/while/lstm_cell/ReadVariableOp_2:value:04rnn_1/while/lstm_cell/strided_slice_2/stack:output:06rnn_1/while/lstm_cell/strided_slice_2/stack_1:output:06rnn_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2'
%rnn_1/while/lstm_cell/strided_slice_2Í
rnn_1/while/lstm_cell/MatMul_6MatMulrnn_1/while/lstm_cell/mul_2:z:0.rnn_1/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_6É
rnn_1/while/lstm_cell/add_2AddV2(rnn_1/while/lstm_cell/BiasAdd_2:output:0(rnn_1/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_2
rnn_1/while/lstm_cell/TanhTanhrnn_1/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/Tanh¶
rnn_1/while/lstm_cell/mul_5Mul!rnn_1/while/lstm_cell/Sigmoid:y:0rnn_1/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_5·
rnn_1/while/lstm_cell/add_3AddV2rnn_1/while/lstm_cell/mul_4:z:0rnn_1/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_3Á
&rnn_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp/rnn_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02(
&rnn_1/while/lstm_cell/ReadVariableOp_3«
+rnn_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2-
+rnn_1/while/lstm_cell/strided_slice_3/stack¯
-rnn_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-rnn_1/while/lstm_cell/strided_slice_3/stack_1¯
-rnn_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-rnn_1/while/lstm_cell/strided_slice_3/stack_2
%rnn_1/while/lstm_cell/strided_slice_3StridedSlice.rnn_1/while/lstm_cell/ReadVariableOp_3:value:04rnn_1/while/lstm_cell/strided_slice_3/stack:output:06rnn_1/while/lstm_cell/strided_slice_3/stack_1:output:06rnn_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2'
%rnn_1/while/lstm_cell/strided_slice_3Í
rnn_1/while/lstm_cell/MatMul_7MatMulrnn_1/while/lstm_cell/mul_3:z:0.rnn_1/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
rnn_1/while/lstm_cell/MatMul_7É
rnn_1/while/lstm_cell/add_4AddV2(rnn_1/while/lstm_cell/BiasAdd_3:output:0(rnn_1/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/add_4 
rnn_1/while/lstm_cell/Sigmoid_2Sigmoidrnn_1/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
rnn_1/while/lstm_cell/Sigmoid_2
rnn_1/while/lstm_cell/Tanh_1Tanhrnn_1/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/Tanh_1º
rnn_1/while/lstm_cell/mul_6Mul#rnn_1/while/lstm_cell/Sigmoid_2:y:0 rnn_1/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/lstm_cell/mul_6
rnn_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
rnn_1/while/Tile/multiples½
rnn_1/while/TileTile8rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0#rnn_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn_1/while/TileÁ
rnn_1/while/SelectV2SelectV2rnn_1/while/Tile:output:0rnn_1/while/lstm_cell/mul_6:z:0rnn_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/SelectV2
rnn_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
rnn_1/while/Tile_1/multiplesÃ
rnn_1/while/Tile_1Tile8rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0%rnn_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn_1/while/Tile_1
rnn_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
rnn_1/while/Tile_2/multiplesÃ
rnn_1/while/Tile_2Tile8rnn_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0%rnn_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn_1/while/Tile_2Ç
rnn_1/while/SelectV2_1SelectV2rnn_1/while/Tile_1:output:0rnn_1/while/lstm_cell/mul_6:z:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/SelectV2_1Ç
rnn_1/while/SelectV2_2SelectV2rnn_1/while/Tile_2:output:0rnn_1/while/lstm_cell/add_3:z:0rnn_1_while_placeholder_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/SelectV2_2ù
0rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_1_while_placeholder_1rnn_1_while_placeholderrnn_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype022
0rnn_1/while/TensorArrayV2Write/TensorListSetItemh
rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add/y
rnn_1/while/addAddV2rnn_1_while_placeholderrnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/addl
rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add_1/y
rnn_1/while/add_1AddV2$rnn_1_while_rnn_1_while_loop_counterrnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/add_1p
rnn_1/while/IdentityIdentityrnn_1/while/add_1:z:0*
T0*
_output_shapes
: 2
rnn_1/while/Identity
rnn_1/while/Identity_1Identity*rnn_1_while_rnn_1_while_maximum_iterations*
T0*
_output_shapes
: 2
rnn_1/while/Identity_1r
rnn_1/while/Identity_2Identityrnn_1/while/add:z:0*
T0*
_output_shapes
: 2
rnn_1/while/Identity_2
rnn_1/while/Identity_3Identity@rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rnn_1/while/Identity_3
rnn_1/while/Identity_4Identityrnn_1/while/SelectV2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/Identity_4
rnn_1/while/Identity_5Identityrnn_1/while/SelectV2_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/Identity_5
rnn_1/while/Identity_6Identityrnn_1/while/SelectV2_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/while/Identity_6"5
rnn_1_while_identityrnn_1/while/Identity:output:0"9
rnn_1_while_identity_1rnn_1/while/Identity_1:output:0"9
rnn_1_while_identity_2rnn_1/while/Identity_2:output:0"9
rnn_1_while_identity_3rnn_1/while/Identity_3:output:0"9
rnn_1_while_identity_4rnn_1/while/Identity_4:output:0"9
rnn_1_while_identity_5rnn_1/while/Identity_5:output:0"9
rnn_1_while_identity_6rnn_1/while/Identity_6:output:0"`
-rnn_1_while_lstm_cell_readvariableop_resource/rnn_1_while_lstm_cell_readvariableop_resource_0"p
5rnn_1_while_lstm_cell_split_1_readvariableop_resource7rnn_1_while_lstm_cell_split_1_readvariableop_resource_0"l
3rnn_1_while_lstm_cell_split_readvariableop_resource5rnn_1_while_lstm_cell_split_readvariableop_resource_0"H
!rnn_1_while_rnn_1_strided_slice_1#rnn_1_while_rnn_1_strided_slice_1_0"È
arnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensorcrnn_1_while_tensorarrayv2read_1_tensorlistgetitem_rnn_1_tensorarrayunstack_1_tensorlistfromtensor_0"À
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*f
_input_shapesU
S: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ís
Ô
while_body_483031
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstÄ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/ones_likep
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02&
$while/lstm_cell/split/ReadVariableOpë
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
while/lstm_cell/split¾
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMulÂ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_1Â
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_2Â
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpß
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell/split_1³
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd¹
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_1¹
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_2¹
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/BiasAdd_3
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul 
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_1 
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_2 
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_3«
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Ü
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
while/lstm_cell/strided_slice±
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_4«
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid¯
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2è
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1µ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_5±
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_4¯
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2è
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2µ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_6±
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_5
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_3¯
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2è
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3µ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/MatMul_7±
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/Tanh_1¢
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/lstm_cell/mul_6Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3}
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_4}
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 


&__inference_rnn_1_layer_call_fn_483736

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4815462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ê·
Ý
F__inference_sequential_layer_call_and_return_conditional_losses_482569

inputs1
-rnn_1_lstm_cell_split_readvariableop_resource3
/rnn_1_lstm_cell_split_1_readvariableop_resource+
'rnn_1_lstm_cell_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢rnn_1/whilem
masking/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
masking/NotEqual/y
masking/NotEqualNotEqualinputsmasking/NotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
masking/NotEqual
masking/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
masking/Any/reduction_indices¦
masking/AnyAnymasking/NotEqual:z:0&masking/Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
masking/Any
masking/CastCastmasking/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
masking/Cast{
masking/mulMulinputsmasking/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
masking/mul
masking/SqueezeSqueezemasking/Any:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
masking/SqueezeY
rnn_1/ShapeShapemasking/mul:z:0*
T0*
_output_shapes
:2
rnn_1/Shape
rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice/stack
rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_1
rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_2
rnn_1/strided_sliceStridedSlicernn_1/Shape:output:0"rnn_1/strided_slice/stack:output:0$rnn_1/strided_slice/stack_1:output:0$rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_sliceh
rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros/mul/y
rnn_1/zeros/mulMulrnn_1/strided_slice:output:0rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/mulk
rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros/Less/y
rnn_1/zeros/LessLessrnn_1/zeros/mul:z:0rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/Lessn
rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros/packed/1
rnn_1/zeros/packedPackrnn_1/strided_slice:output:0rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros/packedk
rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros/Const
rnn_1/zerosFillrnn_1/zeros/packed:output:0rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/zerosl
rnn_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros_1/mul/y
rnn_1/zeros_1/mulMulrnn_1/strided_slice:output:0rnn_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/mulo
rnn_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros_1/Less/y
rnn_1/zeros_1/LessLessrnn_1/zeros_1/mul:z:0rnn_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/Lessr
rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn_1/zeros_1/packed/1¡
rnn_1/zeros_1/packedPackrnn_1/strided_slice:output:0rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros_1/packedo
rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros_1/Const
rnn_1/zeros_1Fillrnn_1/zeros_1/packed:output:0rnn_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/zeros_1
rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose/perm
rnn_1/transpose	Transposemasking/mul:z:0rnn_1/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
rnn_1/transposea
rnn_1/Shape_1Shapernn_1/transpose:y:0*
T0*
_output_shapes
:2
rnn_1/Shape_1
rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_1/stack
rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_1
rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_2
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1:output:0$rnn_1/strided_slice_1/stack:output:0&rnn_1/strided_slice_1/stack_1:output:0&rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_slice_1w
rnn_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rnn_1/ExpandDims/dimª
rnn_1/ExpandDims
ExpandDimsmasking/Squeeze:output:0rnn_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn_1/ExpandDims
rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose_1/perm®
rnn_1/transpose_1	Transposernn_1/ExpandDims:output:0rnn_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn_1/transpose_1
!rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!rnn_1/TensorArrayV2/element_shapeÊ
rnn_1/TensorArrayV2TensorListReserve*rnn_1/TensorArrayV2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2Ë
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2=
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_1/transpose:y:0Drnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-rnn_1/TensorArrayUnstack/TensorListFromTensor
rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_2/stack
rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_1
rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_2¡
rnn_1/strided_slice_2StridedSlicernn_1/transpose:y:0$rnn_1/strided_slice_2/stack:output:0&rnn_1/strided_slice_2/stack_1:output:0&rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
rnn_1/strided_slice_2
rnn_1/lstm_cell/ones_like/ShapeShapernn_1/zeros:output:0*
T0*
_output_shapes
:2!
rnn_1/lstm_cell/ones_like/Shape
rnn_1/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
rnn_1/lstm_cell/ones_like/ConstÄ
rnn_1/lstm_cell/ones_likeFill(rnn_1/lstm_cell/ones_like/Shape:output:0(rnn_1/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/ones_likep
rnn_1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/lstm_cell/Const
rnn_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
rnn_1/lstm_cell/split/split_dim¼
$rnn_1/lstm_cell/split/ReadVariableOpReadVariableOp-rnn_1_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02&
$rnn_1/lstm_cell/split/ReadVariableOpë
rnn_1/lstm_cell/splitSplit(rnn_1/lstm_cell/split/split_dim:output:0,rnn_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
rnn_1/lstm_cell/split¬
rnn_1/lstm_cell/MatMulMatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul°
rnn_1/lstm_cell/MatMul_1MatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_1°
rnn_1/lstm_cell/MatMul_2MatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_2°
rnn_1/lstm_cell/MatMul_3MatMulrnn_1/strided_slice_2:output:0rnn_1/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_3t
rnn_1/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/lstm_cell/Const_1
!rnn_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!rnn_1/lstm_cell/split_1/split_dim½
&rnn_1/lstm_cell/split_1/ReadVariableOpReadVariableOp/rnn_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02(
&rnn_1/lstm_cell/split_1/ReadVariableOpß
rnn_1/lstm_cell/split_1Split*rnn_1/lstm_cell/split_1/split_dim:output:0.rnn_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
rnn_1/lstm_cell/split_1³
rnn_1/lstm_cell/BiasAddBiasAdd rnn_1/lstm_cell/MatMul:product:0 rnn_1/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd¹
rnn_1/lstm_cell/BiasAdd_1BiasAdd"rnn_1/lstm_cell/MatMul_1:product:0 rnn_1/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd_1¹
rnn_1/lstm_cell/BiasAdd_2BiasAdd"rnn_1/lstm_cell/MatMul_2:product:0 rnn_1/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd_2¹
rnn_1/lstm_cell/BiasAdd_3BiasAdd"rnn_1/lstm_cell/MatMul_3:product:0 rnn_1/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/BiasAdd_3
rnn_1/lstm_cell/mulMulrnn_1/zeros:output:0"rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul¡
rnn_1/lstm_cell/mul_1Mulrnn_1/zeros:output:0"rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_1¡
rnn_1/lstm_cell/mul_2Mulrnn_1/zeros:output:0"rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_2¡
rnn_1/lstm_cell/mul_3Mulrnn_1/zeros:output:0"rnn_1/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_3©
rnn_1/lstm_cell/ReadVariableOpReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02 
rnn_1/lstm_cell/ReadVariableOp
#rnn_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#rnn_1/lstm_cell/strided_slice/stack
%rnn_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2'
%rnn_1/lstm_cell/strided_slice/stack_1
%rnn_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%rnn_1/lstm_cell/strided_slice/stack_2Ü
rnn_1/lstm_cell/strided_sliceStridedSlice&rnn_1/lstm_cell/ReadVariableOp:value:0,rnn_1/lstm_cell/strided_slice/stack:output:0.rnn_1/lstm_cell/strided_slice/stack_1:output:0.rnn_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
rnn_1/lstm_cell/strided_slice±
rnn_1/lstm_cell/MatMul_4MatMulrnn_1/lstm_cell/mul:z:0&rnn_1/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_4«
rnn_1/lstm_cell/addAddV2 rnn_1/lstm_cell/BiasAdd:output:0"rnn_1/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add
rnn_1/lstm_cell/SigmoidSigmoidrnn_1/lstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Sigmoid­
 rnn_1/lstm_cell/ReadVariableOp_1ReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 rnn_1/lstm_cell/ReadVariableOp_1
%rnn_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2'
%rnn_1/lstm_cell/strided_slice_1/stack£
'rnn_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'rnn_1/lstm_cell/strided_slice_1/stack_1£
'rnn_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rnn_1/lstm_cell/strided_slice_1/stack_2è
rnn_1/lstm_cell/strided_slice_1StridedSlice(rnn_1/lstm_cell/ReadVariableOp_1:value:0.rnn_1/lstm_cell/strided_slice_1/stack:output:00rnn_1/lstm_cell/strided_slice_1/stack_1:output:00rnn_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
rnn_1/lstm_cell/strided_slice_1µ
rnn_1/lstm_cell/MatMul_5MatMulrnn_1/lstm_cell/mul_1:z:0(rnn_1/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_5±
rnn_1/lstm_cell/add_1AddV2"rnn_1/lstm_cell/BiasAdd_1:output:0"rnn_1/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_1
rnn_1/lstm_cell/Sigmoid_1Sigmoidrnn_1/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Sigmoid_1
rnn_1/lstm_cell/mul_4Mulrnn_1/lstm_cell/Sigmoid_1:y:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_4­
 rnn_1/lstm_cell/ReadVariableOp_2ReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 rnn_1/lstm_cell/ReadVariableOp_2
%rnn_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%rnn_1/lstm_cell/strided_slice_2/stack£
'rnn_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2)
'rnn_1/lstm_cell/strided_slice_2/stack_1£
'rnn_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rnn_1/lstm_cell/strided_slice_2/stack_2è
rnn_1/lstm_cell/strided_slice_2StridedSlice(rnn_1/lstm_cell/ReadVariableOp_2:value:0.rnn_1/lstm_cell/strided_slice_2/stack:output:00rnn_1/lstm_cell/strided_slice_2/stack_1:output:00rnn_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
rnn_1/lstm_cell/strided_slice_2µ
rnn_1/lstm_cell/MatMul_6MatMulrnn_1/lstm_cell/mul_2:z:0(rnn_1/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_6±
rnn_1/lstm_cell/add_2AddV2"rnn_1/lstm_cell/BiasAdd_2:output:0"rnn_1/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_2
rnn_1/lstm_cell/TanhTanhrnn_1/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Tanh
rnn_1/lstm_cell/mul_5Mulrnn_1/lstm_cell/Sigmoid:y:0rnn_1/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_5
rnn_1/lstm_cell/add_3AddV2rnn_1/lstm_cell/mul_4:z:0rnn_1/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_3­
 rnn_1/lstm_cell/ReadVariableOp_3ReadVariableOp'rnn_1_lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 rnn_1/lstm_cell/ReadVariableOp_3
%rnn_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2'
%rnn_1/lstm_cell/strided_slice_3/stack£
'rnn_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rnn_1/lstm_cell/strided_slice_3/stack_1£
'rnn_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rnn_1/lstm_cell/strided_slice_3/stack_2è
rnn_1/lstm_cell/strided_slice_3StridedSlice(rnn_1/lstm_cell/ReadVariableOp_3:value:0.rnn_1/lstm_cell/strided_slice_3/stack:output:00rnn_1/lstm_cell/strided_slice_3/stack_1:output:00rnn_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2!
rnn_1/lstm_cell/strided_slice_3µ
rnn_1/lstm_cell/MatMul_7MatMulrnn_1/lstm_cell/mul_3:z:0(rnn_1/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/MatMul_7±
rnn_1/lstm_cell/add_4AddV2"rnn_1/lstm_cell/BiasAdd_3:output:0"rnn_1/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/add_4
rnn_1/lstm_cell/Sigmoid_2Sigmoidrnn_1/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Sigmoid_2
rnn_1/lstm_cell/Tanh_1Tanhrnn_1/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/Tanh_1¢
rnn_1/lstm_cell/mul_6Mulrnn_1/lstm_cell/Sigmoid_2:y:0rnn_1/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/lstm_cell/mul_6
#rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2%
#rnn_1/TensorArrayV2_1/element_shapeÐ
rnn_1/TensorArrayV2_1TensorListReserve,rnn_1/TensorArrayV2_1/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2_1Z

rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn_1/time
#rnn_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#rnn_1/TensorArrayV2_2/element_shapeÐ
rnn_1/TensorArrayV2_2TensorListReserve,rnn_1/TensorArrayV2_2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
rnn_1/TensorArrayV2_2Ï
=rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=rnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape
/rnn_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorrnn_1/transpose_1:y:0Frnn_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type021
/rnn_1/TensorArrayUnstack_1/TensorListFromTensor~
rnn_1/zeros_like	ZerosLikernn_1/lstm_cell/mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rnn_1/zeros_like
rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
rnn_1/while/maximum_iterationsv
rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/while/loop_counter¸
rnn_1/whileWhile!rnn_1/while/loop_counter:output:0'rnn_1/while/maximum_iterations:output:0rnn_1/time:output:0rnn_1/TensorArrayV2_1:handle:0rnn_1/zeros_like:y:0rnn_1/zeros:output:0rnn_1/zeros_1:output:0rnn_1/strided_slice_1:output:0=rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0?rnn_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0-rnn_1_lstm_cell_split_readvariableop_resource/rnn_1_lstm_cell_split_1_readvariableop_resource'rnn_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *%
_read_only_resource_inputs

*#
bodyR
rnn_1_while_body_482416*#
condR
rnn_1_while_cond_482415*`
output_shapesO
M: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : : *
parallel_iterations 2
rnn_1/whileÁ
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   28
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shape
(rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_1/while:output:3?rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02*
(rnn_1/TensorArrayV2Stack/TensorListStack
rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn_1/strided_slice_3/stack
rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_3/stack_1
rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_3/stack_2¾
rnn_1/strided_slice_3StridedSlice1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_1/strided_slice_3/stack:output:0&rnn_1/strided_slice_3/stack_1:output:0&rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
rnn_1/strided_slice_3
rnn_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose_2/permÆ
rnn_1/transpose_2	Transpose1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0rnn_1/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
rnn_1/transpose_2
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulrnn_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxy
IdentityIdentitydense/Softmax:softmax:0^rnn_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::2
rnn_1/whilernn_1/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
«
Ã
while_cond_483030
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_483030___redundant_placeholder04
0while_while_cond_483030___redundant_placeholder14
0while_while_cond_483030___redundant_placeholder24
0while_while_cond_483030___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:


&__inference_rnn_1_layer_call_fn_483170
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_1_layer_call_and_return_conditional_losses_4810802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
«
Ã
while_cond_483324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_483324___redundant_placeholder04
0while_while_cond_483324___redundant_placeholder14
0while_while_cond_483324___redundant_placeholder24
0while_while_cond_483324___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Ú

__inference__traced_save_484037
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop5
1savev2_rnn_1_lstm_cell_kernel_read_readvariableop?
;savev2_rnn_1_lstm_cell_recurrent_kernel_read_readvariableop3
/savev2_rnn_1_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b469141330e04060a3b4c29c33465960/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameß
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*ñ
valueçBä
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slicesÀ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop1savev2_rnn_1_lstm_cell_kernel_read_readvariableop;savev2_rnn_1_lstm_cell_recurrent_kernel_read_readvariableop/savev2_rnn_1_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*M
_input_shapes<
:: :@::
È:	@:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::&"
 
_output_shapes
:
È:%!

_output_shapes
:	@:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
Â	
_
C__inference_masking_layer_call_and_return_conditional_losses_481235

inputs
identity]

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2

NotEqual/y}
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Any/reduction_indices
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Anyp
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Castc
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
mul
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezei
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ý
­
+__inference_sequential_layer_call_fn_482599

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_4819132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
«
Ã
while_cond_481142
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481142___redundant_placeholder04
0while_while_cond_481142___redundant_placeholder14
0while_while_cond_481142___redundant_placeholder24
0while_while_cond_481142___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
²f

E__inference_lstm_cell_layer_call_and_return_conditional_losses_480643

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeÓ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ç¬è2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeÙ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2ñ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout_1/GreaterEqual/yÆ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeÙ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Ó£ã2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout_2/GreaterEqual/yÆ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeÙ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed±ÿå)*
seed2Æ¼È2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout_3/GreaterEqual/yÆ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Mul_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È*
dtype02
split/ReadVariableOp«
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	BiasAdd_3^
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
muld
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_1d
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ü
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh^
mul_5MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tanh_1b
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_6]
IdentityIdentity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identitya

Identity_1Identity	mul_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
´
ë
A__inference_rnn_1_layer_call_and_return_conditional_losses_483725

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_2t
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const¬
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/ones_liked
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
È*
dtype02 
lstm_cell/split/ReadVariableOpÓ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	È@:	È@:	È@:	È@*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpÇ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd¡
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_1¡
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_2¡
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/BiasAdd_3
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_1
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_2
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2¸
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ä
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_1
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_4
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ä
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_2o
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_5
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ä
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Sigmoid_2s
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/Tanh_1
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lstm_cell/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÛ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_483597*
condR
while_cond_483596*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
transpose_1t
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Â
serving_default®
U
masking_inputD
serving_default_masking_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¬¬
Û%
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
F_default_save_signature
*G&call_and_return_all_conditional_losses
H__call__"µ#
_tf_keras_sequential#{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "masking_input"}}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "mask_value": 0}}, {"class_name": "RNN", "config": {"name": "rnn_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.2, "implementation": 1}}}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 200]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "masking_input"}}, {"class_name": "Masking", "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "mask_value": 0}}, {"class_name": "RNN", "config": {"name": "rnn_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.2, "implementation": 1}}}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.01, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
Ä

	variables
trainable_variables
regularization_losses
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"µ
_tf_keras_layer{"class_name": "Masking", "name": "masking", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "masking", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "mask_value": 0}}

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"ä
_tf_keras_rnn_layerÆ{"class_name": "RNN", "name": "rnn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.2, "implementation": 1}}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 200]}}
î

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"É
_tf_keras_layer¯{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
	variables
trainable_variables
metrics
non_trainable_variables
layer_regularization_losses

 layers
regularization_losses
!layer_metrics
H__call__
F_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Oserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

	variables
trainable_variables
"metrics
#non_trainable_variables
$layer_regularization_losses

%layers
regularization_losses
&layer_metrics
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
¤

kernel
recurrent_kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"é
_tf_keras_layerÏ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.2, "implementation": 1}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹
	variables
trainable_variables
+metrics
,non_trainable_variables
-layer_regularization_losses

.layers
regularization_losses

/states
0layer_metrics
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
:@2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables
1metrics
2non_trainable_variables
3layer_regularization_losses

4layers
regularization_losses
5layer_metrics
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
*:(
È2rnn_1/lstm_cell/kernel
3:1	@2 rnn_1/lstm_cell/recurrent_kernel
#:!2rnn_1/lstm_cell/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
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
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
'	variables
(trainable_variables
8metrics
9non_trainable_variables
:layer_regularization_losses

;layers
)regularization_losses
<layer_metrics
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
»
	=total
	>count
?	variables
@	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	Atotal
	Bcount
C
_fn_kwargs
D	variables
E	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
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
:  (2total
:  (2count
.
=0
>1"
trackable_list_wrapper
-
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
ó2ð
!__inference__wrapped_model_480494Ê
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *:¢7
52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_482288
F__inference_sequential_layer_call_and_return_conditional_losses_482569
F__inference_sequential_layer_call_and_return_conditional_losses_481861
F__inference_sequential_layer_call_and_return_conditional_losses_481844À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_sequential_layer_call_fn_481926
+__inference_sequential_layer_call_fn_482599
+__inference_sequential_layer_call_fn_482584
+__inference_sequential_layer_call_fn_481894À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_masking_layer_call_and_return_conditional_losses_482610¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_masking_layer_call_fn_482615¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
A__inference_rnn_1_layer_call_and_return_conditional_losses_482919
A__inference_rnn_1_layer_call_and_return_conditional_losses_483725
A__inference_rnn_1_layer_call_and_return_conditional_losses_483159
A__inference_rnn_1_layer_call_and_return_conditional_losses_483485æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
&__inference_rnn_1_layer_call_fn_483170
&__inference_rnn_1_layer_call_fn_483736
&__inference_rnn_1_layer_call_fn_483181
&__inference_rnn_1_layer_call_fn_483747æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_483758¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_483767¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
9B7
$__inference_signature_wrapper_481943masking_input
Ò2Ï
E__inference_lstm_cell_layer_call_and_return_conditional_losses_483953
E__inference_lstm_cell_layer_call_and_return_conditional_losses_483876¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_lstm_cell_layer_call_fn_483970
*__inference_lstm_cell_layer_call_fn_483987¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ¡
!__inference__wrapped_model_480494|D¢A
:¢7
52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_layer_call_and_return_conditional_losses_483758\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_layer_call_fn_483767O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÈ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_483876þ¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ@
EB

0/1/0ÿÿÿÿÿÿÿÿÿ@

0/1/1ÿÿÿÿÿÿÿÿÿ@
 È
E__inference_lstm_cell_layer_call_and_return_conditional_losses_483953þ¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ@
EB

0/1/0ÿÿÿÿÿÿÿÿÿ@

0/1/1ÿÿÿÿÿÿÿÿÿ@
 
*__inference_lstm_cell_layer_call_fn_483970î¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ@
A>

1/0ÿÿÿÿÿÿÿÿÿ@

1/1ÿÿÿÿÿÿÿÿÿ@
*__inference_lstm_cell_layer_call_fn_483987î¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ@
A>

1/0ÿÿÿÿÿÿÿÿÿ@

1/1ÿÿÿÿÿÿÿÿÿ@»
C__inference_masking_layer_call_and_return_conditional_losses_482610t=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
(__inference_masking_layer_call_fn_482615g=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈÈ
A__inference_rnn_1_layer_call_and_return_conditional_losses_482919T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 È
A__inference_rnn_1_layer_call_and_return_conditional_losses_483159T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 À
A__inference_rnn_1_layer_call_and_return_conditional_losses_483485{M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 À
A__inference_rnn_1_layer_call_and_return_conditional_losses_483725{M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
&__inference_rnn_1_layer_call_fn_483170uT¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_rnn_1_layer_call_fn_483181uT¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_rnn_1_layer_call_fn_483736nM¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_rnn_1_layer_call_fn_483747nM¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ@Æ
F__inference_sequential_layer_call_and_return_conditional_losses_481844|L¢I
B¢?
52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
F__inference_sequential_layer_call_and_return_conditional_losses_481861|L¢I
B¢?
52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
F__inference_sequential_layer_call_and_return_conditional_losses_482288uE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
F__inference_sequential_layer_call_and_return_conditional_losses_482569uE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_481894oL¢I
B¢?
52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_481926oL¢I
B¢?
52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_482584hE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_482599hE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¶
$__inference_signature_wrapper_481943U¢R
¢ 
KªH
F
masking_input52
masking_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ