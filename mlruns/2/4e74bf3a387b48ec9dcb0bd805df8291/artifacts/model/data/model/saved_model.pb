��0
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
executor_typestring ��
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
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle��element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements#
handle��element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��.
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:Z*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
lstm_25/lstm_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_25/lstm_cell_25/kernel
�
/lstm_25/lstm_cell_25/kernel/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_25/kernel*
_output_shapes
:	�*
dtype0
�
%lstm_25/lstm_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*6
shared_name'%lstm_25/lstm_cell_25/recurrent_kernel
�
9lstm_25/lstm_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_25/lstm_cell_25/recurrent_kernel*
_output_shapes
:	Z�*
dtype0
�
lstm_25/lstm_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_25/lstm_cell_25/bias
�
-lstm_25/lstm_cell_25/bias/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_25/bias*
_output_shapes	
:�*
dtype0
�
lstm_26/lstm_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*,
shared_namelstm_26/lstm_cell_26/kernel
�
/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/kernel*
_output_shapes
:	Z�*
dtype0
�
%lstm_26/lstm_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*6
shared_name'%lstm_26/lstm_cell_26/recurrent_kernel
�
9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_26/lstm_cell_26/recurrent_kernel*
_output_shapes
:	Z�*
dtype0
�
lstm_26/lstm_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_26/lstm_cell_26/bias
�
-lstm_26/lstm_cell_26/bias/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/bias*
_output_shapes	
:�*
dtype0
�
lstm_27/lstm_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*,
shared_namelstm_27/lstm_cell_27/kernel
�
/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/kernel*
_output_shapes
:	Z�*
dtype0
�
%lstm_27/lstm_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*6
shared_name'%lstm_27/lstm_cell_27/recurrent_kernel
�
9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_27/lstm_cell_27/recurrent_kernel*
_output_shapes
:	Z�*
dtype0
�
lstm_27/lstm_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_27/lstm_cell_27/bias
�
-lstm_27/lstm_cell_27/bias/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/bias*
_output_shapes	
:�*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:Z*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
�
"Adam/lstm_25/lstm_cell_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_25/lstm_cell_25/kernel/m
�
6Adam/lstm_25/lstm_cell_25/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_25/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_25/lstm_cell_25/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*=
shared_name.,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m
�
@Adam/lstm_25/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m*
_output_shapes
:	Z�*
dtype0
�
 Adam/lstm_25/lstm_cell_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_25/lstm_cell_25/bias/m
�
4Adam/lstm_25/lstm_cell_25/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_25/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_26/lstm_cell_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*3
shared_name$"Adam/lstm_26/lstm_cell_26/kernel/m
�
6Adam/lstm_26/lstm_cell_26/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_26/lstm_cell_26/kernel/m*
_output_shapes
:	Z�*
dtype0
�
,Adam/lstm_26/lstm_cell_26/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*=
shared_name.,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m
�
@Adam/lstm_26/lstm_cell_26/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m*
_output_shapes
:	Z�*
dtype0
�
 Adam/lstm_26/lstm_cell_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_26/lstm_cell_26/bias/m
�
4Adam/lstm_26/lstm_cell_26/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_26/lstm_cell_26/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_27/lstm_cell_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*3
shared_name$"Adam/lstm_27/lstm_cell_27/kernel/m
�
6Adam/lstm_27/lstm_cell_27/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_27/lstm_cell_27/kernel/m*
_output_shapes
:	Z�*
dtype0
�
,Adam/lstm_27/lstm_cell_27/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*=
shared_name.,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m
�
@Adam/lstm_27/lstm_cell_27/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m*
_output_shapes
:	Z�*
dtype0
�
 Adam/lstm_27/lstm_cell_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_27/lstm_cell_27/bias/m
�
4Adam/lstm_27/lstm_cell_27/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_27/lstm_cell_27/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:Z*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
�
"Adam/lstm_25/lstm_cell_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_25/lstm_cell_25/kernel/v
�
6Adam/lstm_25/lstm_cell_25/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_25/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_25/lstm_cell_25/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*=
shared_name.,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v
�
@Adam/lstm_25/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v*
_output_shapes
:	Z�*
dtype0
�
 Adam/lstm_25/lstm_cell_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_25/lstm_cell_25/bias/v
�
4Adam/lstm_25/lstm_cell_25/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_25/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_26/lstm_cell_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*3
shared_name$"Adam/lstm_26/lstm_cell_26/kernel/v
�
6Adam/lstm_26/lstm_cell_26/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_26/lstm_cell_26/kernel/v*
_output_shapes
:	Z�*
dtype0
�
,Adam/lstm_26/lstm_cell_26/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*=
shared_name.,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v
�
@Adam/lstm_26/lstm_cell_26/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v*
_output_shapes
:	Z�*
dtype0
�
 Adam/lstm_26/lstm_cell_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_26/lstm_cell_26/bias/v
�
4Adam/lstm_26/lstm_cell_26/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_26/lstm_cell_26/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_27/lstm_cell_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*3
shared_name$"Adam/lstm_27/lstm_cell_27/kernel/v
�
6Adam/lstm_27/lstm_cell_27/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_27/lstm_cell_27/kernel/v*
_output_shapes
:	Z�*
dtype0
�
,Adam/lstm_27/lstm_cell_27/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z�*=
shared_name.,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v
�
@Adam/lstm_27/lstm_cell_27/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v*
_output_shapes
:	Z�*
dtype0
�
 Adam/lstm_27/lstm_cell_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_27/lstm_cell_27/bias/v
�
4Adam/lstm_27/lstm_cell_27/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_27/lstm_cell_27/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�I
value�IB�I B�I
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
 regularization_losses
!	keras_api
l
"cell
#
state_spec
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�

2beta_1

3beta_2
	4decay
5learning_rate
6iter,m�-m�7m�8m�9m�:m�;m�<m�=m�>m�?m�,v�-v�7v�8v�9v�:v�;v�<v�=v�>v�?v�
N
70
81
92
:3
;4
<5
=6
>7
?8
,9
-10
N
70
81
92
:3
;4
<5
=6
>7
?8
,9
-10
 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
		variables

trainable_variables
regularization_losses
 
�
E
state_size

7kernel
8recurrent_kernel
9bias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
 

70
81
92

70
81
92
 
�

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
�
U
state_size

:kernel
;recurrent_kernel
<bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
 

:0
;1
<2

:0
;1
<2
 
�

Zstates
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
 regularization_losses
�
e
state_size

=kernel
>recurrent_kernel
?bias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
 

=0
>1
?2

=0
>1
?2
 
�

jstates
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
$	variables
%trainable_variables
&regularization_losses
 
 
 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
�
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
.	variables
/trainable_variables
0regularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_25/lstm_cell_25/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_25/lstm_cell_25/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_25/lstm_cell_25/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_26/lstm_cell_26/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_26/lstm_cell_26/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_26/lstm_cell_26/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_27/lstm_cell_27/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_27/lstm_cell_27/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_27/lstm_cell_27/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6

z0
{1
|2
 
 
 

70
81
92

70
81
92
 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
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
 
 

:0
;1
<2

:0
;1
<2
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
 
 

0
 
 
 
 
 
 
 
 
 

=0
>1
?2

=0
>1
?2
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
 
 

"0
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
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_25/lstm_cell_25/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_25/lstm_cell_25/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_25/lstm_cell_25/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_26/lstm_cell_26/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_26/lstm_cell_26/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_26/lstm_cell_26/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_27/lstm_cell_27/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_27/lstm_cell_27/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_27/lstm_cell_27/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_25/lstm_cell_25/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_25/lstm_cell_25/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_25/lstm_cell_25/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_26/lstm_cell_26/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_26/lstm_cell_26/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_26/lstm_cell_26/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_27/lstm_cell_27/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_27/lstm_cell_27/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_27/lstm_cell_27/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_25_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_25_inputlstm_25/lstm_cell_25/kernel%lstm_25/lstm_cell_25/recurrent_kernellstm_25/lstm_cell_25/biaslstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_136594
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp/lstm_25/lstm_cell_25/kernel/Read/ReadVariableOp9lstm_25/lstm_cell_25/recurrent_kernel/Read/ReadVariableOp-lstm_25/lstm_cell_25/bias/Read/ReadVariableOp/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOp9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOp-lstm_26/lstm_cell_26/bias/Read/ReadVariableOp/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOp9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOp-lstm_27/lstm_cell_27/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp6Adam/lstm_25/lstm_cell_25/kernel/m/Read/ReadVariableOp@Adam/lstm_25/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_25/lstm_cell_25/bias/m/Read/ReadVariableOp6Adam/lstm_26/lstm_cell_26/kernel/m/Read/ReadVariableOp@Adam/lstm_26/lstm_cell_26/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_26/lstm_cell_26/bias/m/Read/ReadVariableOp6Adam/lstm_27/lstm_cell_27/kernel/m/Read/ReadVariableOp@Adam/lstm_27/lstm_cell_27/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_27/lstm_cell_27/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp6Adam/lstm_25/lstm_cell_25/kernel/v/Read/ReadVariableOp@Adam/lstm_25/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_25/lstm_cell_25/bias/v/Read/ReadVariableOp6Adam/lstm_26/lstm_cell_26/kernel/v/Read/ReadVariableOp@Adam/lstm_26/lstm_cell_26/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_26/lstm_cell_26/bias/v/Read/ReadVariableOp6Adam/lstm_27/lstm_cell_27/kernel/v/Read/ReadVariableOp@Adam/lstm_27/lstm_cell_27/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_27/lstm_cell_27/bias/v/Read/ReadVariableOpConst*9
Tin2
02.	*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_139926
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/biasbeta_1beta_2decaylearning_rate	Adam/iterlstm_25/lstm_cell_25/kernel%lstm_25/lstm_cell_25/recurrent_kernellstm_25/lstm_cell_25/biaslstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biastotalcounttotal_1count_1total_2count_2Adam/dense_7/kernel/mAdam/dense_7/bias/m"Adam/lstm_25/lstm_cell_25/kernel/m,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m Adam/lstm_25/lstm_cell_25/bias/m"Adam/lstm_26/lstm_cell_26/kernel/m,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m Adam/lstm_26/lstm_cell_26/bias/m"Adam/lstm_27/lstm_cell_27/kernel/m,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m Adam/lstm_27/lstm_cell_27/bias/mAdam/dense_7/kernel/vAdam/dense_7/bias/v"Adam/lstm_25/lstm_cell_25/kernel/v,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v Adam/lstm_25/lstm_cell_25/bias/v"Adam/lstm_26/lstm_cell_26/kernel/v,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v Adam/lstm_26/lstm_cell_26/bias/v"Adam/lstm_27/lstm_cell_27/kernel/v,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v Adam/lstm_27/lstm_cell_27/bias/v*8
Tin1
/2-*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_140068��-
�
�
&sequential_7_lstm_25_while_cond_133863F
Bsequential_7_lstm_25_while_sequential_7_lstm_25_while_loop_counterL
Hsequential_7_lstm_25_while_sequential_7_lstm_25_while_maximum_iterations*
&sequential_7_lstm_25_while_placeholder,
(sequential_7_lstm_25_while_placeholder_1,
(sequential_7_lstm_25_while_placeholder_2,
(sequential_7_lstm_25_while_placeholder_3H
Dsequential_7_lstm_25_while_less_sequential_7_lstm_25_strided_slice_1^
Zsequential_7_lstm_25_while_sequential_7_lstm_25_while_cond_133863___redundant_placeholder0^
Zsequential_7_lstm_25_while_sequential_7_lstm_25_while_cond_133863___redundant_placeholder1^
Zsequential_7_lstm_25_while_sequential_7_lstm_25_while_cond_133863___redundant_placeholder2^
Zsequential_7_lstm_25_while_sequential_7_lstm_25_while_cond_133863___redundant_placeholder3'
#sequential_7_lstm_25_while_identity
�
sequential_7/lstm_25/while/LessLess&sequential_7_lstm_25_while_placeholderDsequential_7_lstm_25_while_less_sequential_7_lstm_25_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_25/while/IdentityIdentity#sequential_7/lstm_25/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_25_while_identity,sequential_7/lstm_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_lstm_27_layer_call_fn_138859

inputs
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135994o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�"
�
while_body_134666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_26_134690_0:	Z�.
while_lstm_cell_26_134692_0:	Z�*
while_lstm_cell_26_134694_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_26_134690:	Z�,
while_lstm_cell_26_134692:	Z�(
while_lstm_cell_26_134694:	���*while/lstm_cell_26/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_134690_0while_lstm_cell_26_134692_0while_lstm_cell_26_134694_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134652�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������Z�
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������Zy

while/NoOpNoOp+^while/lstm_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_26_134690while_lstm_cell_26_134690_0"8
while_lstm_cell_26_134692while_lstm_cell_26_134692_0"8
while_lstm_cell_26_134694while_lstm_cell_26_134694_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2X
*while/lstm_cell_26/StatefulPartitionedCall*while/lstm_cell_26/StatefulPartitionedCall: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�	
e
F__inference_dropout_27_layer_call_and_return_conditional_losses_135835

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������Zo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������Zi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������ZY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�	
e
F__inference_dropout_27_layer_call_and_return_conditional_losses_139458

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������Zo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������Zi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������ZY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�8
�
while_body_138275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_26_biasadd_readvariableop_resource:	���)while/lstm_cell_26/BiasAdd/ReadVariableOp�(while/lstm_cell_26/MatMul/ReadVariableOp�*while/lstm_cell_26/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0 while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_3Sigmoid!while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_26/Sigmoid_4Sigmoidwhile/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_3:y:0 while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_139771

inputs
states_0
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�
�
-__inference_lstm_cell_26_layer_call_fn_139609

inputs
states_0
states_1
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�

�
-__inference_sequential_7_layer_call_fn_135805
lstm_25_input
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
	unknown_2:	Z�
	unknown_3:	Z�
	unknown_4:	�
	unknown_5:	Z�
	unknown_6:	Z�
	unknown_7:	�
	unknown_8:Z
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_135780o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_25_input
�

�
lstm_27_while_cond_137430,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1D
@lstm_27_while_lstm_27_while_cond_137430___redundant_placeholder0D
@lstm_27_while_lstm_27_while_cond_137430___redundant_placeholder1D
@lstm_27_while_lstm_27_while_cond_137430___redundant_placeholder2D
@lstm_27_while_lstm_27_while_cond_137430___redundant_placeholder3
lstm_27_while_identity
�
lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: [
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_lstm_27_layer_call_fn_138837
inputs_0
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135276o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_27_layer_call_fn_139707

inputs
states_0
states_1
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135148o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�

�
lstm_26_while_cond_137283,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1D
@lstm_26_while_lstm_26_while_cond_137283___redundant_placeholder0D
@lstm_26_while_lstm_26_while_cond_137283___redundant_placeholder1D
@lstm_26_while_lstm_26_while_cond_137283___redundant_placeholder2D
@lstm_26_while_lstm_26_while_cond_137283___redundant_placeholder3
lstm_26_while_identity
�
lstm_26/while/LessLesslstm_26_while_placeholder*lstm_26_while_less_lstm_26_strided_slice_1*
T0*
_output_shapes
: [
lstm_26/while/IdentityIdentitylstm_26/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_26_while_identitylstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�

�
-__inference_sequential_7_layer_call_fn_136621

inputs
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
	unknown_2:	Z�
	unknown_3:	Z�
	unknown_4:	�
	unknown_5:	Z�
	unknown_6:	Z�
	unknown_7:	�
	unknown_8:Z
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_135780o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_25_layer_call_and_return_conditional_losses_138172

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Zs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Zm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�J
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_137716
inputs_0>
+lstm_cell_25_matmul_readvariableop_resource:	�@
-lstm_cell_25_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_25_biasadd_readvariableop_resource:	�
identity��#lstm_cell_25/BiasAdd/ReadVariableOp�"lstm_cell_25/MatMul/ReadVariableOp�$lstm_cell_25/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_3Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_25/Sigmoid_4Sigmoidlstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_3:y:0lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_137632*
condR
while_cond_137631*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Z�
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�8
�
while_body_136286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	�F
3while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_25_biasadd_readvariableop_resource:	���)while/lstm_cell_25/BiasAdd/ReadVariableOp�(while/lstm_cell_25/MatMul/ReadVariableOp�*while/lstm_cell_25/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0 while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_3Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_25/Sigmoid_4Sigmoidwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_3:y:0 while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
d
F__inference_dropout_25_layer_call_and_return_conditional_losses_135447

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������Z_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
while_cond_135663
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_135663___redundant_placeholder04
0while_while_cond_135663___redundant_placeholder14
0while_while_cond_135663___redundant_placeholder24
0while_while_cond_135663___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�

�
-__inference_sequential_7_layer_call_fn_136648

inputs
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
	unknown_2:	Z�
	unknown_3:	Z�
	unknown_4:	�
	unknown_5:	Z�
	unknown_6:	Z�
	unknown_7:	�
	unknown_8:Z
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_136441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_138917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_138917___redundant_placeholder04
0while_while_cond_138917___redundant_placeholder14
0while_while_cond_138917___redundant_placeholder24
0while_while_cond_138917___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�$
�
H__inference_sequential_7_layer_call_and_return_conditional_losses_136441

inputs!
lstm_25_136411:	�!
lstm_25_136413:	Z�
lstm_25_136415:	�!
lstm_26_136419:	Z�!
lstm_26_136421:	Z�
lstm_26_136423:	�!
lstm_27_136427:	Z�!
lstm_27_136429:	Z�
lstm_27_136431:	� 
dense_7_136435:Z
dense_7_136437:
identity��dense_7/StatefulPartitionedCall�"dropout_25/StatefulPartitionedCall�"dropout_26/StatefulPartitionedCall�"dropout_27/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�lstm_26/StatefulPartitionedCall�lstm_27/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCallinputslstm_25_136411lstm_25_136413lstm_25_136415*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_136370�
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_136211�
lstm_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0lstm_26_136419lstm_26_136421lstm_26_136423*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_136182�
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_26_layer_call_and_return_conditional_losses_136023�
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0lstm_27_136427lstm_27_136429lstm_27_136431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135994�
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_26/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_135835�
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_7_136435dense_7_136437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_135773w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_26_layer_call_and_return_conditional_losses_136023

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Zs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Zm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�8
�
while_body_138704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_26_biasadd_readvariableop_resource:	���)while/lstm_cell_26/BiasAdd/ReadVariableOp�(while/lstm_cell_26/MatMul/ReadVariableOp�*while/lstm_cell_26/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0 while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_3Sigmoid!while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_26/Sigmoid_4Sigmoidwhile/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_3:y:0 while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_139446

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������Z[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�A
�

lstm_26_while_body_137284,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�P
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�K
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	Z�N
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�I
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	���1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp�0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp�2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp�
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_3Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_4Sigmoid$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_3:y:0(lstm_26/while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_26_while_placeholder_1lstm_26_while_placeholder$lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_26/while/addAddV2lstm_26_while_placeholderlstm_26/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: �
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: q
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: �
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: �
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_26/while/NoOpNoOp2^lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1^lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp3^lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_26_while_identitylstm_26/while/Identity:output:0"=
lstm_26_while_identity_1!lstm_26/while/Identity_1:output:0"=
lstm_26_while_identity_2!lstm_26/while/Identity_2:output:0"=
lstm_26_while_identity_3!lstm_26/while/Identity_3:output:0"=
lstm_26_while_identity_4!lstm_26/while/Identity_4:output:0"=
lstm_26_while_identity_5!lstm_26/while/Identity_5:output:0"P
%lstm_26_while_lstm_26_strided_slice_1'lstm_26_while_lstm_26_strided_slice_1_0"z
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"|
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"x
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"�
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2f
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2d
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2h
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_lstm_25_layer_call_fn_137540
inputs_0
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_134385|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_139203
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_139203___redundant_placeholder04
0while_while_cond_139203___redundant_placeholder14
0while_while_cond_139203___redundant_placeholder24
0while_while_cond_139203___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�8
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_134385

inputs&
lstm_cell_25_134303:	�&
lstm_cell_25_134305:	Z�"
lstm_cell_25_134307:	�
identity��$lstm_cell_25/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25_134303lstm_cell_25_134305lstm_cell_25_134307*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134302n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25_134303lstm_cell_25_134305lstm_cell_25_134307*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_134316*
condR
while_cond_134315*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Zu
NoOpNoOp%^lstm_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_25/StatefulPartitionedCall$lstm_cell_25/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�8
�
while_body_135910
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_27_biasadd_readvariableop_resource:	���)while/lstm_cell_27/BiasAdd/ReadVariableOp�(while/lstm_cell_27/MatMul/ReadVariableOp�*while/lstm_cell_27/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0 while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_3Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_27/Sigmoid_4Sigmoidwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_3:y:0 while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�

�
$__inference_signature_wrapper_136594
lstm_25_input
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
	unknown_2:	Z�
	unknown_3:	Z�
	unknown_4:	�
	unknown_5:	Z�
	unknown_6:	Z�
	unknown_7:	�
	unknown_8:Z
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_134235o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_25_input
�A
�

lstm_25_while_body_136707,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:	�P
=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�K
<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorL
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:	�N
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�I
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	���1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp�0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp�2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp�
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_25/while/lstm_cell_25/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
#lstm_25/while/lstm_cell_25/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_25/while/lstm_cell_25/addAddV2+lstm_25/while/lstm_cell_25/MatMul:product:0-lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_25/while/lstm_cell_25/BiasAddBiasAdd"lstm_25/while/lstm_cell_25/add:z:09lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_25/while/lstm_cell_25/splitSplit3lstm_25/while/lstm_cell_25/split/split_dim:output:0+lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
"lstm_25/while/lstm_cell_25/SigmoidSigmoid)lstm_25/while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_25/while/lstm_cell_25/mulMul(lstm_25/while/lstm_cell_25/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
 lstm_25/while/lstm_cell_25/mul_1Mul&lstm_25/while/lstm_cell_25/Sigmoid:y:0(lstm_25/while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
 lstm_25/while/lstm_cell_25/add_1AddV2"lstm_25/while/lstm_cell_25/mul:z:0$lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_3Sigmoid)lstm_25/while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_4Sigmoid$lstm_25/while/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
 lstm_25/while/lstm_cell_25/mul_2Mul(lstm_25/while/lstm_cell_25/Sigmoid_3:y:0(lstm_25/while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_25_while_placeholder_1lstm_25_while_placeholder$lstm_25/while/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_25/while/addAddV2lstm_25_while_placeholderlstm_25/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_25/while/add_1AddV2(lstm_25_while_lstm_25_while_loop_counterlstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_25/while/IdentityIdentitylstm_25/while/add_1:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: �
lstm_25/while/Identity_1Identity.lstm_25_while_lstm_25_while_maximum_iterations^lstm_25/while/NoOp*
T0*
_output_shapes
: q
lstm_25/while/Identity_2Identitylstm_25/while/add:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: �
lstm_25/while/Identity_3IdentityBlstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_25/while/NoOp*
T0*
_output_shapes
: �
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_25/mul_2:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_25/add_1:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_25/while/NoOpNoOp2^lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1^lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp3^lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_25_while_identitylstm_25/while/Identity:output:0"=
lstm_25_while_identity_1!lstm_25/while/Identity_1:output:0"=
lstm_25_while_identity_2!lstm_25/while/Identity_2:output:0"=
lstm_25_while_identity_3!lstm_25/while/Identity_3:output:0"=
lstm_25_while_identity_4!lstm_25/while/Identity_4:output:0"=
lstm_25_while_identity_5!lstm_25/while/Identity_5:output:0"P
%lstm_25_while_lstm_25_strided_slice_1'lstm_25_while_lstm_25_strided_slice_1_0"z
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0"|
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0"x
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0"�
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2f
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp2d
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp2h
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_134857
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_26_134881_0:	Z�.
while_lstm_cell_26_134883_0:	Z�*
while_lstm_cell_26_134885_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_26_134881:	Z�,
while_lstm_cell_26_134883:	Z�(
while_lstm_cell_26_134885:	���*while/lstm_cell_26/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_134881_0while_lstm_cell_26_134883_0while_lstm_cell_26_134885_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134798�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������Z�
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������Zy

while/NoOpNoOp+^while/lstm_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_26_134881while_lstm_cell_26_134881_0"8
while_lstm_cell_26_134883while_lstm_cell_26_134883_0"8
while_lstm_cell_26_134885while_lstm_cell_26_134885_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2X
*while/lstm_cell_26/StatefulPartitionedCall*while/lstm_cell_26/StatefulPartitionedCall: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_135761

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������Z[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�J
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_139431

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	Z�@
-lstm_cell_27_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_27_biasadd_readvariableop_resource:	�
identity��#lstm_cell_27/BiasAdd/ReadVariableOp�"lstm_cell_27/MatMul/ReadVariableOp�$lstm_cell_27/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_3Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_27/Sigmoid_4Sigmoidlstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_3:y:0lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_139347*
condR
while_cond_139346*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
(__inference_lstm_27_layer_call_fn_138848

inputs
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
G
+__inference_dropout_25_layer_call_fn_138150

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_135447d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�P
�
&sequential_7_lstm_26_while_body_134004F
Bsequential_7_lstm_26_while_sequential_7_lstm_26_while_loop_counterL
Hsequential_7_lstm_26_while_sequential_7_lstm_26_while_maximum_iterations*
&sequential_7_lstm_26_while_placeholder,
(sequential_7_lstm_26_while_placeholder_1,
(sequential_7_lstm_26_while_placeholder_2,
(sequential_7_lstm_26_while_placeholder_3E
Asequential_7_lstm_26_while_sequential_7_lstm_26_strided_slice_1_0�
}sequential_7_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_26_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_7_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�]
Jsequential_7_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�X
Isequential_7_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	�'
#sequential_7_lstm_26_while_identity)
%sequential_7_lstm_26_while_identity_1)
%sequential_7_lstm_26_while_identity_2)
%sequential_7_lstm_26_while_identity_3)
%sequential_7_lstm_26_while_identity_4)
%sequential_7_lstm_26_while_identity_5C
?sequential_7_lstm_26_while_sequential_7_lstm_26_strided_slice_1
{sequential_7_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_26_tensorarrayunstack_tensorlistfromtensorY
Fsequential_7_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	Z�[
Hsequential_7_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�V
Gsequential_7_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	���>sequential_7/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp�=sequential_7/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp�?sequential_7/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp�
Lsequential_7/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
>sequential_7/lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_26_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_26_while_placeholderUsequential_7/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
=sequential_7/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
.sequential_7/lstm_26/while/lstm_cell_26/MatMulMatMulEsequential_7/lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
?sequential_7/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
0sequential_7/lstm_26/while/lstm_cell_26/MatMul_1MatMul(sequential_7_lstm_26_while_placeholder_2Gsequential_7/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_7/lstm_26/while/lstm_cell_26/addAddV28sequential_7/lstm_26/while/lstm_cell_26/MatMul:product:0:sequential_7/lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
>sequential_7/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
/sequential_7/lstm_26/while/lstm_cell_26/BiasAddBiasAdd/sequential_7/lstm_26/while/lstm_cell_26/add:z:0Fsequential_7/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
7sequential_7/lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
-sequential_7/lstm_26/while/lstm_cell_26/splitSplit@sequential_7/lstm_26/while/lstm_cell_26/split/split_dim:output:08sequential_7/lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
/sequential_7/lstm_26/while/lstm_cell_26/SigmoidSigmoid6sequential_7/lstm_26/while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid6sequential_7/lstm_26/while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_26/while/lstm_cell_26/mulMul5sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_1:y:0(sequential_7_lstm_26_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid6sequential_7/lstm_26/while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_26/while/lstm_cell_26/mul_1Mul3sequential_7/lstm_26/while/lstm_cell_26/Sigmoid:y:05sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_26/while/lstm_cell_26/add_1AddV2/sequential_7/lstm_26/while/lstm_cell_26/mul:z:01sequential_7/lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_3Sigmoid6sequential_7/lstm_26/while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_4Sigmoid1sequential_7/lstm_26/while/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_26/while/lstm_cell_26/mul_2Mul5sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_3:y:05sequential_7/lstm_26/while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
?sequential_7/lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_26_while_placeholder_1&sequential_7_lstm_26_while_placeholder1sequential_7/lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���b
 sequential_7/lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_7/lstm_26/while/addAddV2&sequential_7_lstm_26_while_placeholder)sequential_7/lstm_26/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_7/lstm_26/while/add_1AddV2Bsequential_7_lstm_26_while_sequential_7_lstm_26_while_loop_counter+sequential_7/lstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: �
#sequential_7/lstm_26/while/IdentityIdentity$sequential_7/lstm_26/while/add_1:z:0 ^sequential_7/lstm_26/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_26/while/Identity_1IdentityHsequential_7_lstm_26_while_sequential_7_lstm_26_while_maximum_iterations ^sequential_7/lstm_26/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_26/while/Identity_2Identity"sequential_7/lstm_26/while/add:z:0 ^sequential_7/lstm_26/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_26/while/Identity_3IdentityOsequential_7/lstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_26/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_26/while/Identity_4Identity1sequential_7/lstm_26/while/lstm_cell_26/mul_2:z:0 ^sequential_7/lstm_26/while/NoOp*
T0*'
_output_shapes
:���������Z�
%sequential_7/lstm_26/while/Identity_5Identity1sequential_7/lstm_26/while/lstm_cell_26/add_1:z:0 ^sequential_7/lstm_26/while/NoOp*
T0*'
_output_shapes
:���������Z�
sequential_7/lstm_26/while/NoOpNoOp?^sequential_7/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp>^sequential_7/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp@^sequential_7/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_26_while_identity,sequential_7/lstm_26/while/Identity:output:0"W
%sequential_7_lstm_26_while_identity_1.sequential_7/lstm_26/while/Identity_1:output:0"W
%sequential_7_lstm_26_while_identity_2.sequential_7/lstm_26/while/Identity_2:output:0"W
%sequential_7_lstm_26_while_identity_3.sequential_7/lstm_26/while/Identity_3:output:0"W
%sequential_7_lstm_26_while_identity_4.sequential_7/lstm_26/while/Identity_4:output:0"W
%sequential_7_lstm_26_while_identity_5.sequential_7/lstm_26/while/Identity_5:output:0"�
Gsequential_7_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resourceIsequential_7_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"�
Hsequential_7_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resourceJsequential_7_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"�
Fsequential_7_lstm_26_while_lstm_cell_26_matmul_readvariableop_resourceHsequential_7_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"�
?sequential_7_lstm_26_while_sequential_7_lstm_26_strided_slice_1Asequential_7_lstm_26_while_sequential_7_lstm_26_strided_slice_1_0"�
{sequential_7_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_26_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2�
>sequential_7/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp>sequential_7/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp=sequential_7/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2�
?sequential_7/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp?sequential_7/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
G
+__inference_dropout_27_layer_call_fn_139436

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_135761`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
� 
�
H__inference_sequential_7_layer_call_and_return_conditional_losses_135780

inputs!
lstm_25_135435:	�!
lstm_25_135437:	Z�
lstm_25_135439:	�!
lstm_26_135592:	Z�!
lstm_26_135594:	Z�
lstm_26_135596:	�!
lstm_27_135749:	Z�!
lstm_27_135751:	Z�
lstm_27_135753:	� 
dense_7_135774:Z
dense_7_135776:
identity��dense_7/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�lstm_26/StatefulPartitionedCall�lstm_27/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCallinputslstm_25_135435lstm_25_135437lstm_25_135439*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_135434�
dropout_25/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_135447�
lstm_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0lstm_26_135592lstm_26_135594lstm_26_135596*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_135591�
dropout_26/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_26_layer_call_and_return_conditional_losses_135604�
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0lstm_27_135749lstm_27_135751lstm_27_135753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135748�
dropout_27/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_135761�
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_7_135774dense_7_135776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_135773w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_7/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_137917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_137917___redundant_placeholder04
0while_while_cond_137917___redundant_placeholder14
0while_while_cond_137917___redundant_placeholder24
0while_while_cond_137917___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�J
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_136370

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	�@
-lstm_cell_25_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_25_biasadd_readvariableop_resource:	�
identity��#lstm_cell_25/BiasAdd/ReadVariableOp�"lstm_cell_25/MatMul/ReadVariableOp�$lstm_cell_25/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_3Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_25/Sigmoid_4Sigmoidlstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_3:y:0lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_136286*
condR
while_cond_136285*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_135506
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_135506___redundant_placeholder04
0while_while_cond_135506___redundant_placeholder14
0while_while_cond_135506___redundant_placeholder24
0while_while_cond_135506___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_134506
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_134506___redundant_placeholder04
0while_while_cond_134506___redundant_placeholder14
0while_while_cond_134506___redundant_placeholder24
0while_while_cond_134506___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_138560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_138560___redundant_placeholder04
0while_while_cond_138560___redundant_placeholder14
0while_while_cond_138560___redundant_placeholder24
0while_while_cond_138560___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�J
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_135748

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	Z�@
-lstm_cell_27_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_27_biasadd_readvariableop_resource:	�
identity��#lstm_cell_27/BiasAdd/ReadVariableOp�"lstm_cell_27/MatMul/ReadVariableOp�$lstm_cell_27/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_3Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_27/Sigmoid_4Sigmoidlstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_3:y:0lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_135664*
condR
while_cond_135663*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�

�
lstm_25_while_cond_136706,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1D
@lstm_25_while_lstm_25_while_cond_136706___redundant_placeholder0D
@lstm_25_while_lstm_25_while_cond_136706___redundant_placeholder1D
@lstm_25_while_lstm_25_while_cond_136706___redundant_placeholder2D
@lstm_25_while_lstm_25_while_cond_136706___redundant_placeholder3
lstm_25_while_identity
�
lstm_25/while/LessLesslstm_25_while_placeholder*lstm_25_while_less_lstm_25_strided_slice_1*
T0*
_output_shapes
: [
lstm_25/while/IdentityIdentitylstm_25/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_25_while_identitylstm_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_138417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_138417___redundant_placeholder04
0while_while_cond_138417___redundant_placeholder14
0while_while_cond_138417___redundant_placeholder24
0while_while_cond_138417___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�8
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_134576

inputs&
lstm_cell_25_134494:	�&
lstm_cell_25_134496:	Z�"
lstm_cell_25_134498:	�
identity��$lstm_cell_25/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25_134494lstm_cell_25_134496lstm_cell_25_134498*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134448n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25_134494lstm_cell_25_134496lstm_cell_25_134498*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_134507*
condR
while_cond_134506*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Zu
NoOpNoOp%^lstm_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_25/StatefulPartitionedCall$lstm_cell_25/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�8
�
while_body_137918
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	�F
3while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_25_biasadd_readvariableop_resource:	���)while/lstm_cell_25/BiasAdd/ReadVariableOp�(while/lstm_cell_25/MatMul/ReadVariableOp�*while/lstm_cell_25/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0 while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_3Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_25/Sigmoid_4Sigmoidwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_3:y:0 while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
d
F__inference_dropout_26_layer_call_and_return_conditional_losses_138803

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������Z_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_140068
file_prefix1
assignvariableop_dense_7_kernel:Z-
assignvariableop_1_dense_7_bias:#
assignvariableop_2_beta_1: #
assignvariableop_3_beta_2: "
assignvariableop_4_decay: *
 assignvariableop_5_learning_rate: &
assignvariableop_6_adam_iter:	 A
.assignvariableop_7_lstm_25_lstm_cell_25_kernel:	�K
8assignvariableop_8_lstm_25_lstm_cell_25_recurrent_kernel:	Z�;
,assignvariableop_9_lstm_25_lstm_cell_25_bias:	�B
/assignvariableop_10_lstm_26_lstm_cell_26_kernel:	Z�L
9assignvariableop_11_lstm_26_lstm_cell_26_recurrent_kernel:	Z�<
-assignvariableop_12_lstm_26_lstm_cell_26_bias:	�B
/assignvariableop_13_lstm_27_lstm_cell_27_kernel:	Z�L
9assignvariableop_14_lstm_27_lstm_cell_27_recurrent_kernel:	Z�<
-assignvariableop_15_lstm_27_lstm_cell_27_bias:	�#
assignvariableop_16_total: #
assignvariableop_17_count: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: %
assignvariableop_20_total_2: %
assignvariableop_21_count_2: ;
)assignvariableop_22_adam_dense_7_kernel_m:Z5
'assignvariableop_23_adam_dense_7_bias_m:I
6assignvariableop_24_adam_lstm_25_lstm_cell_25_kernel_m:	�S
@assignvariableop_25_adam_lstm_25_lstm_cell_25_recurrent_kernel_m:	Z�C
4assignvariableop_26_adam_lstm_25_lstm_cell_25_bias_m:	�I
6assignvariableop_27_adam_lstm_26_lstm_cell_26_kernel_m:	Z�S
@assignvariableop_28_adam_lstm_26_lstm_cell_26_recurrent_kernel_m:	Z�C
4assignvariableop_29_adam_lstm_26_lstm_cell_26_bias_m:	�I
6assignvariableop_30_adam_lstm_27_lstm_cell_27_kernel_m:	Z�S
@assignvariableop_31_adam_lstm_27_lstm_cell_27_recurrent_kernel_m:	Z�C
4assignvariableop_32_adam_lstm_27_lstm_cell_27_bias_m:	�;
)assignvariableop_33_adam_dense_7_kernel_v:Z5
'assignvariableop_34_adam_dense_7_bias_v:I
6assignvariableop_35_adam_lstm_25_lstm_cell_25_kernel_v:	�S
@assignvariableop_36_adam_lstm_25_lstm_cell_25_recurrent_kernel_v:	Z�C
4assignvariableop_37_adam_lstm_25_lstm_cell_25_bias_v:	�I
6assignvariableop_38_adam_lstm_26_lstm_cell_26_kernel_v:	Z�S
@assignvariableop_39_adam_lstm_26_lstm_cell_26_recurrent_kernel_v:	Z�C
4assignvariableop_40_adam_lstm_26_lstm_cell_26_bias_v:	�I
6assignvariableop_41_adam_lstm_27_lstm_cell_27_kernel_v:	Z�S
@assignvariableop_42_adam_lstm_27_lstm_cell_27_recurrent_kernel_v:	Z�C
4assignvariableop_43_adam_lstm_27_lstm_cell_27_bias_v:	�
identity_45��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_beta_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_beta_2Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_25_lstm_cell_25_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_25_lstm_cell_25_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_25_lstm_cell_25_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_26_lstm_cell_26_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_26_lstm_cell_26_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_26_lstm_cell_26_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_lstm_27_lstm_cell_27_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp9assignvariableop_14_lstm_27_lstm_cell_27_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_lstm_27_lstm_cell_27_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_7_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_7_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_25_lstm_cell_25_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_25_lstm_cell_25_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_25_lstm_cell_25_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_26_lstm_cell_26_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_26_lstm_cell_26_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_26_lstm_cell_26_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_27_lstm_cell_27_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_27_lstm_cell_27_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_27_lstm_cell_27_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_7_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_7_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_lstm_25_lstm_cell_25_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp@assignvariableop_36_adam_lstm_25_lstm_cell_25_recurrent_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_lstm_25_lstm_cell_25_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_lstm_26_lstm_cell_26_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_lstm_26_lstm_cell_26_recurrent_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_lstm_26_lstm_cell_26_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_lstm_27_lstm_cell_27_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp@assignvariableop_42_adam_lstm_27_lstm_cell_27_recurrent_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_lstm_27_lstm_cell_27_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_45IdentityIdentity_44:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
while_cond_136097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_136097___redundant_placeholder04
0while_while_cond_136097___redundant_placeholder14
0while_while_cond_136097___redundant_placeholder24
0while_while_cond_136097___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�8
�
while_body_138561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_26_biasadd_readvariableop_resource:	���)while/lstm_cell_26/BiasAdd/ReadVariableOp�(while/lstm_cell_26/MatMul/ReadVariableOp�*while/lstm_cell_26/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0 while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_3Sigmoid!while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_26/Sigmoid_4Sigmoidwhile/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_3:y:0 while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�J
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_138145

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	�@
-lstm_cell_25_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_25_biasadd_readvariableop_resource:	�
identity��#lstm_cell_25/BiasAdd/ReadVariableOp�"lstm_cell_25/MatMul/ReadVariableOp�$lstm_cell_25/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_3Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_25/Sigmoid_4Sigmoidlstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_3:y:0lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_138061*
condR
while_cond_138060*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_135085

inputs&
lstm_cell_27_135003:	Z�&
lstm_cell_27_135005:	Z�"
lstm_cell_27_135007:	�
identity��$lstm_cell_27/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_135003lstm_cell_27_135005lstm_cell_27_135007*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135002n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_135003lstm_cell_27_135005lstm_cell_27_135007*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_135016*
condR
while_cond_135015*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Zu
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������Z
 
_user_specified_nameinputs
�8
�
while_body_136098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_26_biasadd_readvariableop_resource:	���)while/lstm_cell_26/BiasAdd/ReadVariableOp�(while/lstm_cell_26/MatMul/ReadVariableOp�*while/lstm_cell_26/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0 while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_3Sigmoid!while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_26/Sigmoid_4Sigmoidwhile/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_3:y:0 while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_135350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	�F
3while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_25_biasadd_readvariableop_resource:	���)while/lstm_cell_25/BiasAdd/ReadVariableOp�(while/lstm_cell_25/MatMul/ReadVariableOp�*while/lstm_cell_25/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0 while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_3Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_25/Sigmoid_4Sigmoidwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_3:y:0 while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�J
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_139145
inputs_0>
+lstm_cell_27_matmul_readvariableop_resource:	Z�@
-lstm_cell_27_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_27_biasadd_readvariableop_resource:	�
identity��#lstm_cell_27/BiasAdd/ReadVariableOp�"lstm_cell_27/MatMul/ReadVariableOp�$lstm_cell_27/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_3Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_27/Sigmoid_4Sigmoidlstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_3:y:0lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_139061*
condR
while_cond_139060*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�
�
while_cond_138060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_138060___redundant_placeholder04
0while_while_cond_138060___redundant_placeholder14
0while_while_cond_138060___redundant_placeholder24
0while_while_cond_138060___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�P
�
&sequential_7_lstm_25_while_body_133864F
Bsequential_7_lstm_25_while_sequential_7_lstm_25_while_loop_counterL
Hsequential_7_lstm_25_while_sequential_7_lstm_25_while_maximum_iterations*
&sequential_7_lstm_25_while_placeholder,
(sequential_7_lstm_25_while_placeholder_1,
(sequential_7_lstm_25_while_placeholder_2,
(sequential_7_lstm_25_while_placeholder_3E
Asequential_7_lstm_25_while_sequential_7_lstm_25_strided_slice_1_0�
}sequential_7_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_25_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_7_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:	�]
Jsequential_7_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�X
Isequential_7_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	�'
#sequential_7_lstm_25_while_identity)
%sequential_7_lstm_25_while_identity_1)
%sequential_7_lstm_25_while_identity_2)
%sequential_7_lstm_25_while_identity_3)
%sequential_7_lstm_25_while_identity_4)
%sequential_7_lstm_25_while_identity_5C
?sequential_7_lstm_25_while_sequential_7_lstm_25_strided_slice_1
{sequential_7_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_25_tensorarrayunstack_tensorlistfromtensorY
Fsequential_7_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:	�[
Hsequential_7_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�V
Gsequential_7_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	���>sequential_7/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp�=sequential_7/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp�?sequential_7/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp�
Lsequential_7/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_7/lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_25_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_25_while_placeholderUsequential_7/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
=sequential_7/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
.sequential_7/lstm_25/while/lstm_cell_25/MatMulMatMulEsequential_7/lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
?sequential_7/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
0sequential_7/lstm_25/while/lstm_cell_25/MatMul_1MatMul(sequential_7_lstm_25_while_placeholder_2Gsequential_7/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_7/lstm_25/while/lstm_cell_25/addAddV28sequential_7/lstm_25/while/lstm_cell_25/MatMul:product:0:sequential_7/lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
>sequential_7/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
/sequential_7/lstm_25/while/lstm_cell_25/BiasAddBiasAdd/sequential_7/lstm_25/while/lstm_cell_25/add:z:0Fsequential_7/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
7sequential_7/lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
-sequential_7/lstm_25/while/lstm_cell_25/splitSplit@sequential_7/lstm_25/while/lstm_cell_25/split/split_dim:output:08sequential_7/lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
/sequential_7/lstm_25/while/lstm_cell_25/SigmoidSigmoid6sequential_7/lstm_25/while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid6sequential_7/lstm_25/while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_25/while/lstm_cell_25/mulMul5sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_1:y:0(sequential_7_lstm_25_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid6sequential_7/lstm_25/while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_25/while/lstm_cell_25/mul_1Mul3sequential_7/lstm_25/while/lstm_cell_25/Sigmoid:y:05sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_25/while/lstm_cell_25/add_1AddV2/sequential_7/lstm_25/while/lstm_cell_25/mul:z:01sequential_7/lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_3Sigmoid6sequential_7/lstm_25/while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_4Sigmoid1sequential_7/lstm_25/while/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_25/while/lstm_cell_25/mul_2Mul5sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_3:y:05sequential_7/lstm_25/while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
?sequential_7/lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_25_while_placeholder_1&sequential_7_lstm_25_while_placeholder1sequential_7/lstm_25/while/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���b
 sequential_7/lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_7/lstm_25/while/addAddV2&sequential_7_lstm_25_while_placeholder)sequential_7/lstm_25/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_7/lstm_25/while/add_1AddV2Bsequential_7_lstm_25_while_sequential_7_lstm_25_while_loop_counter+sequential_7/lstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: �
#sequential_7/lstm_25/while/IdentityIdentity$sequential_7/lstm_25/while/add_1:z:0 ^sequential_7/lstm_25/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_25/while/Identity_1IdentityHsequential_7_lstm_25_while_sequential_7_lstm_25_while_maximum_iterations ^sequential_7/lstm_25/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_25/while/Identity_2Identity"sequential_7/lstm_25/while/add:z:0 ^sequential_7/lstm_25/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_25/while/Identity_3IdentityOsequential_7/lstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_25/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_25/while/Identity_4Identity1sequential_7/lstm_25/while/lstm_cell_25/mul_2:z:0 ^sequential_7/lstm_25/while/NoOp*
T0*'
_output_shapes
:���������Z�
%sequential_7/lstm_25/while/Identity_5Identity1sequential_7/lstm_25/while/lstm_cell_25/add_1:z:0 ^sequential_7/lstm_25/while/NoOp*
T0*'
_output_shapes
:���������Z�
sequential_7/lstm_25/while/NoOpNoOp?^sequential_7/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp>^sequential_7/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp@^sequential_7/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_25_while_identity,sequential_7/lstm_25/while/Identity:output:0"W
%sequential_7_lstm_25_while_identity_1.sequential_7/lstm_25/while/Identity_1:output:0"W
%sequential_7_lstm_25_while_identity_2.sequential_7/lstm_25/while/Identity_2:output:0"W
%sequential_7_lstm_25_while_identity_3.sequential_7/lstm_25/while/Identity_3:output:0"W
%sequential_7_lstm_25_while_identity_4.sequential_7/lstm_25/while/Identity_4:output:0"W
%sequential_7_lstm_25_while_identity_5.sequential_7/lstm_25/while/Identity_5:output:0"�
Gsequential_7_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resourceIsequential_7_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0"�
Hsequential_7_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resourceJsequential_7_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0"�
Fsequential_7_lstm_25_while_lstm_cell_25_matmul_readvariableop_resourceHsequential_7_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0"�
?sequential_7_lstm_25_while_sequential_7_lstm_25_strided_slice_1Asequential_7_lstm_25_while_sequential_7_lstm_25_strided_slice_1_0"�
{sequential_7_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_25_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2�
>sequential_7/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp>sequential_7/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp=sequential_7/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp2�
?sequential_7/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp?sequential_7/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�A
�

lstm_26_while_body_136847,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�P
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�K
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	Z�N
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�I
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	���1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp�0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp�2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp�
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_3Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Z�
$lstm_26/while/lstm_cell_26/Sigmoid_4Sigmoid$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_3:y:0(lstm_26/while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_26_while_placeholder_1lstm_26_while_placeholder$lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_26/while/addAddV2lstm_26_while_placeholderlstm_26/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: �
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: q
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: �
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: �
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_26/while/NoOpNoOp2^lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1^lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp3^lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_26_while_identitylstm_26/while/Identity:output:0"=
lstm_26_while_identity_1!lstm_26/while/Identity_1:output:0"=
lstm_26_while_identity_2!lstm_26/while/Identity_2:output:0"=
lstm_26_while_identity_3!lstm_26/while/Identity_3:output:0"=
lstm_26_while_identity_4!lstm_26/while/Identity_4:output:0"=
lstm_26_while_identity_5!lstm_26/while/Identity_5:output:0"P
%lstm_26_while_lstm_26_strided_slice_1'lstm_26_while_lstm_26_strided_slice_1_0"z
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"|
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"x
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"�
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2f
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2d
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2h
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134798

inputs

states
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates
�
G
+__inference_dropout_26_layer_call_fn_138793

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_26_layer_call_and_return_conditional_losses_135604d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�"
�
while_body_135016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_27_135040_0:	Z�.
while_lstm_cell_27_135042_0:	Z�*
while_lstm_cell_27_135044_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_27_135040:	Z�,
while_lstm_cell_27_135042:	Z�(
while_lstm_cell_27_135044:	���*while/lstm_cell_27/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_135040_0while_lstm_cell_27_135042_0while_lstm_cell_27_135044_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135002�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������Z�
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������Zy

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_27_135040while_lstm_cell_27_135040_0"8
while_lstm_cell_27_135042while_lstm_cell_27_135042_0"8
while_lstm_cell_27_135044while_lstm_cell_27_135044_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_137632
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	�F
3while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_25_biasadd_readvariableop_resource:	���)while/lstm_cell_25/BiasAdd/ReadVariableOp�(while/lstm_cell_25/MatMul/ReadVariableOp�*while/lstm_cell_25/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0 while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_3Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_25/Sigmoid_4Sigmoidwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_3:y:0 while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�8
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_134926

inputs&
lstm_cell_26_134844:	Z�&
lstm_cell_26_134846:	Z�"
lstm_cell_26_134848:	�
identity��$lstm_cell_26/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_134844lstm_cell_26_134846lstm_cell_26_134848*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134798n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_134844lstm_cell_26_134846lstm_cell_26_134848*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_134857*
condR
while_cond_134856*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Zu
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������Z
 
_user_specified_nameinputs
�
d
F__inference_dropout_26_layer_call_and_return_conditional_losses_135604

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������Z_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
(__inference_lstm_26_layer_call_fn_138205

inputs
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_135591s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
&sequential_7_lstm_26_while_cond_134003F
Bsequential_7_lstm_26_while_sequential_7_lstm_26_while_loop_counterL
Hsequential_7_lstm_26_while_sequential_7_lstm_26_while_maximum_iterations*
&sequential_7_lstm_26_while_placeholder,
(sequential_7_lstm_26_while_placeholder_1,
(sequential_7_lstm_26_while_placeholder_2,
(sequential_7_lstm_26_while_placeholder_3H
Dsequential_7_lstm_26_while_less_sequential_7_lstm_26_strided_slice_1^
Zsequential_7_lstm_26_while_sequential_7_lstm_26_while_cond_134003___redundant_placeholder0^
Zsequential_7_lstm_26_while_sequential_7_lstm_26_while_cond_134003___redundant_placeholder1^
Zsequential_7_lstm_26_while_sequential_7_lstm_26_while_cond_134003___redundant_placeholder2^
Zsequential_7_lstm_26_while_sequential_7_lstm_26_while_cond_134003___redundant_placeholder3'
#sequential_7_lstm_26_while_identity
�
sequential_7/lstm_26/while/LessLess&sequential_7_lstm_26_while_placeholderDsequential_7_lstm_26_while_less_sequential_7_lstm_26_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_26/while/IdentityIdentity#sequential_7/lstm_26/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_26_while_identity,sequential_7/lstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�	
�
C__inference_dense_7_layer_call_and_return_conditional_losses_135773

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
(__inference_lstm_26_layer_call_fn_138216

inputs
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_136182s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
while_cond_139060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_139060___redundant_placeholder04
0while_while_cond_139060___redundant_placeholder14
0while_while_cond_139060___redundant_placeholder24
0while_while_cond_139060___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�J
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_138788

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	Z�@
-lstm_cell_26_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_26_biasadd_readvariableop_resource:	�
identity��#lstm_cell_26/BiasAdd/ReadVariableOp�"lstm_cell_26/MatMul/ReadVariableOp�$lstm_cell_26/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_3Sigmoidlstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_26/Sigmoid_4Sigmoidlstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_3:y:0lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_138704*
condR
while_cond_138703*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
(__inference_lstm_26_layer_call_fn_138194
inputs_0
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_134926|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�8
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_134735

inputs&
lstm_cell_26_134653:	Z�&
lstm_cell_26_134655:	Z�"
lstm_cell_26_134657:	�
identity��$lstm_cell_26/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_134653lstm_cell_26_134655lstm_cell_26_134657*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134652n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_134653lstm_cell_26_134655lstm_cell_26_134657*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_134666*
condR
while_cond_134665*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Zu
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������Z
 
_user_specified_nameinputs
�"
�
while_body_135207
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_27_135231_0:	Z�.
while_lstm_cell_27_135233_0:	Z�*
while_lstm_cell_27_135235_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_27_135231:	Z�,
while_lstm_cell_27_135233:	Z�(
while_lstm_cell_27_135235:	���*while/lstm_cell_27/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_135231_0while_lstm_cell_27_135233_0while_lstm_cell_27_135235_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135148�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������Z�
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������Zy

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_27_135231while_lstm_cell_27_135231_0"8
while_lstm_cell_27_135233while_lstm_cell_27_135233_0"8
while_lstm_cell_27_135235while_lstm_cell_27_135235_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_27_layer_call_fn_139690

inputs
states_0
states_1
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�
�
&sequential_7_lstm_27_while_cond_134143F
Bsequential_7_lstm_27_while_sequential_7_lstm_27_while_loop_counterL
Hsequential_7_lstm_27_while_sequential_7_lstm_27_while_maximum_iterations*
&sequential_7_lstm_27_while_placeholder,
(sequential_7_lstm_27_while_placeholder_1,
(sequential_7_lstm_27_while_placeholder_2,
(sequential_7_lstm_27_while_placeholder_3H
Dsequential_7_lstm_27_while_less_sequential_7_lstm_27_strided_slice_1^
Zsequential_7_lstm_27_while_sequential_7_lstm_27_while_cond_134143___redundant_placeholder0^
Zsequential_7_lstm_27_while_sequential_7_lstm_27_while_cond_134143___redundant_placeholder1^
Zsequential_7_lstm_27_while_sequential_7_lstm_27_while_cond_134143___redundant_placeholder2^
Zsequential_7_lstm_27_while_sequential_7_lstm_27_while_cond_134143___redundant_placeholder3'
#sequential_7_lstm_27_while_identity
�
sequential_7/lstm_27/while/LessLess&sequential_7_lstm_27_while_placeholderDsequential_7_lstm_27_while_less_sequential_7_lstm_27_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_27/while/IdentityIdentity#sequential_7/lstm_27/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_27_while_identity,sequential_7/lstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_dense_7_layer_call_fn_139467

inputs
unknown:Z
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_135773o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
d
F__inference_dropout_25_layer_call_and_return_conditional_losses_138160

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������Z_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�8
�
while_body_135664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_27_biasadd_readvariableop_resource:	���)while/lstm_cell_27/BiasAdd/ReadVariableOp�(while/lstm_cell_27/MatMul/ReadVariableOp�*while/lstm_cell_27/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0 while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_3Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_27/Sigmoid_4Sigmoidwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_3:y:0 while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_139346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_139346___redundant_placeholder04
0while_while_cond_139346___redundant_placeholder14
0while_while_cond_139346___redundant_placeholder24
0while_while_cond_139346___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
d
+__inference_dropout_27_layer_call_fn_139441

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_135835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134448

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates
�

e
F__inference_dropout_26_layer_call_and_return_conditional_losses_138815

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Zs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Zm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�A
�

lstm_27_while_body_136987,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�P
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�K
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorL
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:	Z�N
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�I
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	���1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp�0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp�2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp�
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_3Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_4Sigmoid$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_3:y:0(lstm_27/while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1lstm_27_while_placeholder$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: �
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: q
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: �
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: �
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"�
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�

�
lstm_27_while_cond_136986,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1D
@lstm_27_while_lstm_27_while_cond_136986___redundant_placeholder0D
@lstm_27_while_lstm_27_while_cond_136986___redundant_placeholder1D
@lstm_27_while_lstm_27_while_cond_136986___redundant_placeholder2D
@lstm_27_while_lstm_27_while_cond_136986___redundant_placeholder3
lstm_27_while_identity
�
lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: [
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_135909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_135909___redundant_placeholder04
0while_while_cond_135909___redundant_placeholder14
0while_while_cond_135909___redundant_placeholder24
0while_while_cond_135909___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_136285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_136285___redundant_placeholder04
0while_while_cond_136285___redundant_placeholder14
0while_while_cond_136285___redundant_placeholder24
0while_while_cond_136285___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�

�
lstm_25_while_cond_137136,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1D
@lstm_25_while_lstm_25_while_cond_137136___redundant_placeholder0D
@lstm_25_while_lstm_25_while_cond_137136___redundant_placeholder1D
@lstm_25_while_lstm_25_while_cond_137136___redundant_placeholder2D
@lstm_25_while_lstm_25_while_cond_137136___redundant_placeholder3
lstm_25_while_identity
�
lstm_25/while/LessLesslstm_25_while_placeholder*lstm_25_while_less_lstm_25_strided_slice_1*
T0*
_output_shapes
: [
lstm_25/while/IdentityIdentitylstm_25/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_25_while_identitylstm_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�J
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_139288

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	Z�@
-lstm_cell_27_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_27_biasadd_readvariableop_resource:	�
identity��#lstm_cell_27/BiasAdd/ReadVariableOp�"lstm_cell_27/MatMul/ReadVariableOp�$lstm_cell_27/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_3Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_27/Sigmoid_4Sigmoidlstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_3:y:0lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_139204*
condR
while_cond_139203*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�J
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_138502
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	Z�@
-lstm_cell_26_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_26_biasadd_readvariableop_resource:	�
identity��#lstm_cell_26/BiasAdd/ReadVariableOp�"lstm_cell_26/MatMul/ReadVariableOp�$lstm_cell_26/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_3Sigmoidlstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_26/Sigmoid_4Sigmoidlstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_3:y:0lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_138418*
condR
while_cond_138417*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Z�
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�P
�
&sequential_7_lstm_27_while_body_134144F
Bsequential_7_lstm_27_while_sequential_7_lstm_27_while_loop_counterL
Hsequential_7_lstm_27_while_sequential_7_lstm_27_while_maximum_iterations*
&sequential_7_lstm_27_while_placeholder,
(sequential_7_lstm_27_while_placeholder_1,
(sequential_7_lstm_27_while_placeholder_2,
(sequential_7_lstm_27_while_placeholder_3E
Asequential_7_lstm_27_while_sequential_7_lstm_27_strided_slice_1_0�
}sequential_7_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_27_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_7_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�]
Jsequential_7_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�X
Isequential_7_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	�'
#sequential_7_lstm_27_while_identity)
%sequential_7_lstm_27_while_identity_1)
%sequential_7_lstm_27_while_identity_2)
%sequential_7_lstm_27_while_identity_3)
%sequential_7_lstm_27_while_identity_4)
%sequential_7_lstm_27_while_identity_5C
?sequential_7_lstm_27_while_sequential_7_lstm_27_strided_slice_1
{sequential_7_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_27_tensorarrayunstack_tensorlistfromtensorY
Fsequential_7_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:	Z�[
Hsequential_7_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�V
Gsequential_7_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	���>sequential_7/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp�=sequential_7/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp�?sequential_7/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp�
Lsequential_7/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
>sequential_7/lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_27_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_27_while_placeholderUsequential_7/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
=sequential_7/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
.sequential_7/lstm_27/while/lstm_cell_27/MatMulMatMulEsequential_7/lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
?sequential_7/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
0sequential_7/lstm_27/while/lstm_cell_27/MatMul_1MatMul(sequential_7_lstm_27_while_placeholder_2Gsequential_7/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_7/lstm_27/while/lstm_cell_27/addAddV28sequential_7/lstm_27/while/lstm_cell_27/MatMul:product:0:sequential_7/lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
>sequential_7/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
/sequential_7/lstm_27/while/lstm_cell_27/BiasAddBiasAdd/sequential_7/lstm_27/while/lstm_cell_27/add:z:0Fsequential_7/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
7sequential_7/lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
-sequential_7/lstm_27/while/lstm_cell_27/splitSplit@sequential_7/lstm_27/while/lstm_cell_27/split/split_dim:output:08sequential_7/lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
/sequential_7/lstm_27/while/lstm_cell_27/SigmoidSigmoid6sequential_7/lstm_27/while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid6sequential_7/lstm_27/while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_27/while/lstm_cell_27/mulMul5sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_1:y:0(sequential_7_lstm_27_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid6sequential_7/lstm_27/while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_27/while/lstm_cell_27/mul_1Mul3sequential_7/lstm_27/while/lstm_cell_27/Sigmoid:y:05sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_27/while/lstm_cell_27/add_1AddV2/sequential_7/lstm_27/while/lstm_cell_27/mul:z:01sequential_7/lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_3Sigmoid6sequential_7/lstm_27/while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Z�
1sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_4Sigmoid1sequential_7/lstm_27/while/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
-sequential_7/lstm_27/while/lstm_cell_27/mul_2Mul5sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_3:y:05sequential_7/lstm_27/while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
?sequential_7/lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_27_while_placeholder_1&sequential_7_lstm_27_while_placeholder1sequential_7/lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���b
 sequential_7/lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_7/lstm_27/while/addAddV2&sequential_7_lstm_27_while_placeholder)sequential_7/lstm_27/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_7/lstm_27/while/add_1AddV2Bsequential_7_lstm_27_while_sequential_7_lstm_27_while_loop_counter+sequential_7/lstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: �
#sequential_7/lstm_27/while/IdentityIdentity$sequential_7/lstm_27/while/add_1:z:0 ^sequential_7/lstm_27/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_27/while/Identity_1IdentityHsequential_7_lstm_27_while_sequential_7_lstm_27_while_maximum_iterations ^sequential_7/lstm_27/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_27/while/Identity_2Identity"sequential_7/lstm_27/while/add:z:0 ^sequential_7/lstm_27/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_27/while/Identity_3IdentityOsequential_7/lstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_27/while/NoOp*
T0*
_output_shapes
: �
%sequential_7/lstm_27/while/Identity_4Identity1sequential_7/lstm_27/while/lstm_cell_27/mul_2:z:0 ^sequential_7/lstm_27/while/NoOp*
T0*'
_output_shapes
:���������Z�
%sequential_7/lstm_27/while/Identity_5Identity1sequential_7/lstm_27/while/lstm_cell_27/add_1:z:0 ^sequential_7/lstm_27/while/NoOp*
T0*'
_output_shapes
:���������Z�
sequential_7/lstm_27/while/NoOpNoOp?^sequential_7/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp>^sequential_7/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp@^sequential_7/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_27_while_identity,sequential_7/lstm_27/while/Identity:output:0"W
%sequential_7_lstm_27_while_identity_1.sequential_7/lstm_27/while/Identity_1:output:0"W
%sequential_7_lstm_27_while_identity_2.sequential_7/lstm_27/while/Identity_2:output:0"W
%sequential_7_lstm_27_while_identity_3.sequential_7/lstm_27/while/Identity_3:output:0"W
%sequential_7_lstm_27_while_identity_4.sequential_7/lstm_27/while/Identity_4:output:0"W
%sequential_7_lstm_27_while_identity_5.sequential_7/lstm_27/while/Identity_5:output:0"�
Gsequential_7_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resourceIsequential_7_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"�
Hsequential_7_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resourceJsequential_7_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"�
Fsequential_7_lstm_27_while_lstm_cell_27_matmul_readvariableop_resourceHsequential_7_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"�
?sequential_7_lstm_27_while_sequential_7_lstm_27_strided_slice_1Asequential_7_lstm_27_while_sequential_7_lstm_27_strided_slice_1_0"�
{sequential_7_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_27_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2�
>sequential_7/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp>sequential_7/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp=sequential_7/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2�
?sequential_7/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp?sequential_7/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134652

inputs

states
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates
�8
�
while_body_138918
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_27_biasadd_readvariableop_resource:	���)while/lstm_cell_27/BiasAdd/ReadVariableOp�(while/lstm_cell_27/MatMul/ReadVariableOp�*while/lstm_cell_27/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0 while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_3Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_27/Sigmoid_4Sigmoidwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_3:y:0 while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_134315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_134315___redundant_placeholder04
0while_while_cond_134315___redundant_placeholder14
0while_while_cond_134315___redundant_placeholder24
0while_while_cond_134315___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�8
�
while_body_139347
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_27_biasadd_readvariableop_resource:	���)while/lstm_cell_27/BiasAdd/ReadVariableOp�(while/lstm_cell_27/MatMul/ReadVariableOp�*while/lstm_cell_27/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0 while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_3Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_27/Sigmoid_4Sigmoidwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_3:y:0 while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
��
�

H__inference_sequential_7_layer_call_and_return_conditional_losses_137078

inputsF
3lstm_25_lstm_cell_25_matmul_readvariableop_resource:	�H
5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:	Z�C
4lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	�F
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	Z�H
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:	Z�C
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	�F
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:	Z�H
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:	Z�C
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	�8
&dense_7_matmul_readvariableop_resource:Z5
'dense_7_biasadd_readvariableop_resource:
identity��dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp�*lstm_25/lstm_cell_25/MatMul/ReadVariableOp�,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp�lstm_25/while�+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp�*lstm_26/lstm_cell_26/MatMul/ReadVariableOp�,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp�lstm_26/while�+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp�*lstm_27/lstm_cell_27/MatMul/ReadVariableOp�,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp�lstm_27/whileC
lstm_25/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_sliceStridedSlicelstm_25/Shape:output:0$lstm_25/strided_slice/stack:output:0&lstm_25/strided_slice/stack_1:output:0&lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_25/zeros/packedPacklstm_25/strided_slice:output:0lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:���������ZZ
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_25/zeros_1/packedPacklstm_25/strided_slice:output:0!lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zk
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_25/transpose	Transposeinputslstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_25/Shape_1Shapelstm_25/transpose:y:0*
T0*
_output_shapes
:g
lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_slice_1StridedSlicelstm_25/Shape_1:output:0&lstm_25/strided_slice_1/stack:output:0(lstm_25/strided_slice_1/stack_1:output:0(lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_25/TensorArrayV2TensorListReserve,lstm_25/TensorArrayV2/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_25/transpose:y:0Flstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_25/lstm_cell_25/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_25/lstm_cell_25/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_25/lstm_cell_25/addAddV2%lstm_25/lstm_cell_25/MatMul:product:0'lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_25/lstm_cell_25/BiasAddBiasAddlstm_25/lstm_cell_25/add:z:03lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_25/lstm_cell_25/splitSplit-lstm_25/lstm_cell_25/split/split_dim:output:0%lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split~
lstm_25/lstm_cell_25/SigmoidSigmoid#lstm_25/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/Sigmoid_1Sigmoid#lstm_25/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/mulMul"lstm_25/lstm_cell_25/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/Sigmoid_2Sigmoid#lstm_25/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/mul_1Mul lstm_25/lstm_cell_25/Sigmoid:y:0"lstm_25/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/add_1AddV2lstm_25/lstm_cell_25/mul:z:0lstm_25/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/Sigmoid_3Sigmoid#lstm_25/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Z{
lstm_25/lstm_cell_25/Sigmoid_4Sigmoidlstm_25/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/mul_2Mul"lstm_25/lstm_cell_25/Sigmoid_3:y:0"lstm_25/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zv
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
lstm_25/TensorArrayV2_1TensorListReserve.lstm_25/TensorArrayV2_1/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_25_matmul_readvariableop_resource5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_25_while_body_136707*%
condR
lstm_25_while_cond_136706*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0p
lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskm
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zc
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
dropout_25/IdentityIdentitylstm_25/transpose_1:y:0*
T0*+
_output_shapes
:���������ZY
lstm_26/ShapeShapedropout_25/Identity:output:0*
T0*
_output_shapes
:e
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_sliceStridedSlicelstm_26/Shape:output:0$lstm_26/strided_slice/stack:output:0&lstm_26/strided_slice/stack_1:output:0&lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_26/zeros/packedPacklstm_26/strided_slice:output:0lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*'
_output_shapes
:���������ZZ
lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_26/zeros_1/packedPacklstm_26/strided_slice:output:0!lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zk
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_26/transpose	Transposedropout_25/Identity:output:0lstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:���������ZT
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:g
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split~
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0"lstm_26/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/Sigmoid_3Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Z{
lstm_26/lstm_cell_26/Sigmoid_4Sigmoidlstm_26/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_3:y:0"lstm_26/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zv
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
lstm_26/TensorArrayV2_1TensorListReserve.lstm_26/TensorArrayV2_1/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_26_while_body_136847*%
condR
lstm_26_while_cond_136846*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0p
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskm
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zc
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
dropout_26/IdentityIdentitylstm_26/transpose_1:y:0*
T0*+
_output_shapes
:���������ZY
lstm_27/ShapeShapedropout_26/Identity:output:0*
T0*
_output_shapes
:e
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*'
_output_shapes
:���������ZZ
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zk
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_27/transpose	Transposedropout_26/Identity:output:0lstm_27/transpose/perm:output:0*
T0*+
_output_shapes
:���������ZT
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:g
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split~
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0"lstm_27/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/Sigmoid_3Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Z{
lstm_27/lstm_cell_27/Sigmoid_4Sigmoidlstm_27/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_3:y:0"lstm_27/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zv
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_27_while_body_136987*%
condR
lstm_27_while_cond_136986*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0p
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskm
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zc
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_27/IdentityIdentity lstm_27/strided_slice_3:output:0*
T0*'
_output_shapes
:���������Z�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0�
dense_7/MatMulMatMuldropout_27/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp,^lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+^lstm_25/lstm_cell_25/MatMul/ReadVariableOp-^lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp^lstm_25/while,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2Z
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp2X
*lstm_25/lstm_cell_25/MatMul/ReadVariableOp*lstm_25/lstm_cell_25/MatMul/ReadVariableOp2\
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp2
lstm_25/whilelstm_25/while2Z
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2X
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp*lstm_26/lstm_cell_26/MatMul/ReadVariableOp2\
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2
lstm_26/whilelstm_26/while2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_139575

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�

�
lstm_26_while_cond_136846,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1D
@lstm_26_while_lstm_26_while_cond_136846___redundant_placeholder0D
@lstm_26_while_lstm_26_while_cond_136846___redundant_placeholder1D
@lstm_26_while_lstm_26_while_cond_136846___redundant_placeholder2D
@lstm_26_while_lstm_26_while_cond_136846___redundant_placeholder3
lstm_26_while_identity
�
lstm_26/while/LessLesslstm_26_while_placeholder*lstm_26_while_less_lstm_26_strided_slice_1*
T0*
_output_shapes
: [
lstm_26/while/IdentityIdentitylstm_26/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_26_while_identitylstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�A
�

lstm_27_while_body_137431,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�P
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�K
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorL
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:	Z�N
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�I
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	���1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp�0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp�2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp�
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_3Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Z�
$lstm_27/while/lstm_cell_27/Sigmoid_4Sigmoid$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_3:y:0(lstm_27/while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1lstm_27_while_placeholder$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: �
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: q
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: �
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: �
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"�
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_lstm_25_layer_call_fn_137562

inputs
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_135434s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_138359
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	Z�@
-lstm_cell_26_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_26_biasadd_readvariableop_resource:	�
identity��#lstm_cell_26/BiasAdd/ReadVariableOp�"lstm_cell_26/MatMul/ReadVariableOp�$lstm_cell_26/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_3Sigmoidlstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_26/Sigmoid_4Sigmoidlstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_3:y:0lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_138275*
condR
while_cond_138274*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Z�
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�J
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_135591

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	Z�@
-lstm_cell_26_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_26_biasadd_readvariableop_resource:	�
identity��#lstm_cell_26/BiasAdd/ReadVariableOp�"lstm_cell_26/MatMul/ReadVariableOp�$lstm_cell_26/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_3Sigmoidlstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_26/Sigmoid_4Sigmoidlstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_3:y:0lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_135507*
condR
while_cond_135506*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
(__inference_lstm_27_layer_call_fn_138826
inputs_0
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�
�
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_139673

inputs
states_0
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�
�
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135148

inputs

states
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates
�
�
while_cond_135206
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_135206___redundant_placeholder04
0while_while_cond_135206___redundant_placeholder14
0while_while_cond_135206___redundant_placeholder24
0while_while_cond_135206___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
-__inference_lstm_cell_26_layer_call_fn_139592

inputs
states_0
states_1
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_134652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�[
�
__inference__traced_save_139926
file_prefix-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	:
6savev2_lstm_25_lstm_cell_25_kernel_read_readvariableopD
@savev2_lstm_25_lstm_cell_25_recurrent_kernel_read_readvariableop8
4savev2_lstm_25_lstm_cell_25_bias_read_readvariableop:
6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableopD
@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop8
4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop:
6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableopD
@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop8
4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableopA
=savev2_adam_lstm_25_lstm_cell_25_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_25_lstm_cell_25_bias_m_read_readvariableopA
=savev2_adam_lstm_26_lstm_cell_26_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_26_lstm_cell_26_bias_m_read_readvariableopA
=savev2_adam_lstm_27_lstm_cell_27_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_27_lstm_cell_27_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableopA
=savev2_adam_lstm_25_lstm_cell_25_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_25_lstm_cell_25_bias_v_read_readvariableopA
=savev2_adam_lstm_26_lstm_cell_26_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_26_lstm_cell_26_bias_v_read_readvariableopA
=savev2_adam_lstm_27_lstm_cell_27_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_27_lstm_cell_27_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop6savev2_lstm_25_lstm_cell_25_kernel_read_readvariableop@savev2_lstm_25_lstm_cell_25_recurrent_kernel_read_readvariableop4savev2_lstm_25_lstm_cell_25_bias_read_readvariableop6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableop@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableop@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop=savev2_adam_lstm_25_lstm_cell_25_kernel_m_read_readvariableopGsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_25_lstm_cell_25_bias_m_read_readvariableop=savev2_adam_lstm_26_lstm_cell_26_kernel_m_read_readvariableopGsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_26_lstm_cell_26_bias_m_read_readvariableop=savev2_adam_lstm_27_lstm_cell_27_kernel_m_read_readvariableopGsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_27_lstm_cell_27_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop=savev2_adam_lstm_25_lstm_cell_25_kernel_v_read_readvariableopGsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_25_lstm_cell_25_bias_v_read_readvariableop=savev2_adam_lstm_26_lstm_cell_26_kernel_v_read_readvariableopGsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_26_lstm_cell_26_bias_v_read_readvariableop=savev2_adam_lstm_27_lstm_cell_27_kernel_v_read_readvariableopGsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_27_lstm_cell_27_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :Z:: : : : : :	�:	Z�:�:	Z�:	Z�:�:	Z�:	Z�:�: : : : : : :Z::	�:	Z�:�:	Z�:	Z�:�:	Z�:	Z�:�:Z::	�:	Z�:�:	Z�:	Z�:�:	Z�:	Z�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Z: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	Z�:!


_output_shapes	
:�:%!

_output_shapes
:	Z�:%!

_output_shapes
:	Z�:!

_output_shapes	
:�:%!

_output_shapes
:	Z�:%!

_output_shapes
:	Z�:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	Z�:!

_output_shapes	
:�:%!

_output_shapes
:	Z�:%!

_output_shapes
:	Z�:!

_output_shapes	
:�:%!

_output_shapes
:	Z�:% !

_output_shapes
:	Z�:!!

_output_shapes	
:�:$" 

_output_shapes

:Z: #

_output_shapes
::%$!

_output_shapes
:	�:%%!

_output_shapes
:	Z�:!&

_output_shapes	
:�:%'!

_output_shapes
:	Z�:%(!

_output_shapes
:	Z�:!)

_output_shapes	
:�:%*!

_output_shapes
:	Z�:%+!

_output_shapes
:	Z�:!,

_output_shapes	
:�:-

_output_shapes
: 
�J
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_138002

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	�@
-lstm_cell_25_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_25_biasadd_readvariableop_resource:	�
identity��#lstm_cell_25/BiasAdd/ReadVariableOp�"lstm_cell_25/MatMul/ReadVariableOp�$lstm_cell_25/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_3Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_25/Sigmoid_4Sigmoidlstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_3:y:0lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_137918*
condR
while_cond_137917*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_138061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	�F
3while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_25_biasadd_readvariableop_resource:	���)while/lstm_cell_25/BiasAdd/ReadVariableOp�(while/lstm_cell_25/MatMul/ReadVariableOp�*while/lstm_cell_25/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0 while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_3Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_25/Sigmoid_4Sigmoidwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_3:y:0 while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�J
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_136182

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	Z�@
-lstm_cell_26_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_26_biasadd_readvariableop_resource:	�
identity��#lstm_cell_26/BiasAdd/ReadVariableOp�"lstm_cell_26/MatMul/ReadVariableOp�$lstm_cell_26/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_3Sigmoidlstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_26/Sigmoid_4Sigmoidlstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_3:y:0lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_136098*
condR
while_cond_136097*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
(__inference_lstm_26_layer_call_fn_138183
inputs_0
unknown:	Z�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_134735|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�	
�
C__inference_dense_7_layer_call_and_return_conditional_losses_139477

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�J
�
C__inference_lstm_26_layer_call_and_return_conditional_losses_138645

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	Z�@
-lstm_cell_26_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_26_biasadd_readvariableop_resource:	�
identity��#lstm_cell_26/BiasAdd/ReadVariableOp�"lstm_cell_26/MatMul/ReadVariableOp�$lstm_cell_26/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_26/Sigmoid_3Sigmoidlstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_26/Sigmoid_4Sigmoidlstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_3:y:0lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_138561*
condR
while_cond_138560*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�J
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_137859
inputs_0>
+lstm_cell_25_matmul_readvariableop_resource:	�@
-lstm_cell_25_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_25_biasadd_readvariableop_resource:	�
identity��#lstm_cell_25/BiasAdd/ReadVariableOp�"lstm_cell_25/MatMul/ReadVariableOp�$lstm_cell_25/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_3Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_25/Sigmoid_4Sigmoidlstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_3:y:0lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_137775*
condR
while_cond_137774*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������Z�
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_25_layer_call_fn_139511

inputs
states_0
states_1
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������Z:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�
�
while_cond_135015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_135015___redundant_placeholder04
0while_while_cond_135015___redundant_placeholder14
0while_while_cond_135015___redundant_placeholder24
0while_while_cond_135015___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�8
�
while_body_135507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_26_biasadd_readvariableop_resource:	���)while/lstm_cell_26/BiasAdd/ReadVariableOp�(while/lstm_cell_26/MatMul/ReadVariableOp�*while/lstm_cell_26/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0 while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_3Sigmoid!while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_26/Sigmoid_4Sigmoidwhile/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_3:y:0 while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134302

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates
�
�
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_139543

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�J
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_135434

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	�@
-lstm_cell_25_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_25_biasadd_readvariableop_resource:	�
identity��#lstm_cell_25/BiasAdd/ReadVariableOp�"lstm_cell_25/MatMul/ReadVariableOp�$lstm_cell_25/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_25/Sigmoid_3Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_25/Sigmoid_4Sigmoidlstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_3:y:0lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_135350*
condR
while_cond_135349*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_25_layer_call_and_return_conditional_losses_136211

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Zs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Zm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
while_cond_137631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_137631___redundant_placeholder04
0while_while_cond_137631___redundant_placeholder14
0while_while_cond_137631___redundant_placeholder24
0while_while_cond_137631___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_138703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_138703___redundant_placeholder04
0while_while_cond_138703___redundant_placeholder14
0while_while_cond_138703___redundant_placeholder24
0while_while_cond_138703___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�

�
-__inference_sequential_7_layer_call_fn_136493
lstm_25_input
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
	unknown_2:	Z�
	unknown_3:	Z�
	unknown_4:	�
	unknown_5:	Z�
	unknown_6:	Z�
	unknown_7:	�
	unknown_8:Z
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_136441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_25_input
�8
�
while_body_137775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	�F
3while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_25_biasadd_readvariableop_resource:	���)while/lstm_cell_25/BiasAdd/ReadVariableOp�(while/lstm_cell_25/MatMul/ReadVariableOp�*while/lstm_cell_25/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0 while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_25/Sigmoid_3Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_25/Sigmoid_4Sigmoidwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_3:y:0 while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�8
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_135276

inputs&
lstm_cell_27_135194:	Z�&
lstm_cell_27_135196:	Z�"
lstm_cell_27_135198:	�
identity��$lstm_cell_27/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_135194lstm_cell_27_135196lstm_cell_27_135198*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135148n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_135194lstm_cell_27_135196lstm_cell_27_135198*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_135207*
condR
while_cond_135206*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Zu
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������Z
 
_user_specified_nameinputs
�A
�

lstm_25_while_body_137137,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:	�P
=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:	Z�K
<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	�
lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorL
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:	�N
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:	Z�I
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	���1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp�0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp�2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp�
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_25/while/lstm_cell_25/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
#lstm_25/while/lstm_cell_25/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_25/while/lstm_cell_25/addAddV2+lstm_25/while/lstm_cell_25/MatMul:product:0-lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_25/while/lstm_cell_25/BiasAddBiasAdd"lstm_25/while/lstm_cell_25/add:z:09lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_25/while/lstm_cell_25/splitSplit3lstm_25/while/lstm_cell_25/split/split_dim:output:0+lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
"lstm_25/while/lstm_cell_25/SigmoidSigmoid)lstm_25/while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_25/while/lstm_cell_25/mulMul(lstm_25/while/lstm_cell_25/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
 lstm_25/while/lstm_cell_25/mul_1Mul&lstm_25/while/lstm_cell_25/Sigmoid:y:0(lstm_25/while/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
 lstm_25/while/lstm_cell_25/add_1AddV2"lstm_25/while/lstm_cell_25/mul:z:0$lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_3Sigmoid)lstm_25/while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Z�
$lstm_25/while/lstm_cell_25/Sigmoid_4Sigmoid$lstm_25/while/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
 lstm_25/while/lstm_cell_25/mul_2Mul(lstm_25/while/lstm_cell_25/Sigmoid_3:y:0(lstm_25/while/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_25_while_placeholder_1lstm_25_while_placeholder$lstm_25/while/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_25/while/addAddV2lstm_25_while_placeholderlstm_25/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_25/while/add_1AddV2(lstm_25_while_lstm_25_while_loop_counterlstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_25/while/IdentityIdentitylstm_25/while/add_1:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: �
lstm_25/while/Identity_1Identity.lstm_25_while_lstm_25_while_maximum_iterations^lstm_25/while/NoOp*
T0*
_output_shapes
: q
lstm_25/while/Identity_2Identitylstm_25/while/add:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: �
lstm_25/while/Identity_3IdentityBlstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_25/while/NoOp*
T0*
_output_shapes
: �
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_25/mul_2:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_25/add_1:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:���������Z�
lstm_25/while/NoOpNoOp2^lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1^lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp3^lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_25_while_identitylstm_25/while/Identity:output:0"=
lstm_25_while_identity_1!lstm_25/while/Identity_1:output:0"=
lstm_25_while_identity_2!lstm_25/while/Identity_2:output:0"=
lstm_25_while_identity_3!lstm_25/while/Identity_3:output:0"=
lstm_25_while_identity_4!lstm_25/while/Identity_4:output:0"=
lstm_25_while_identity_5!lstm_25/while/Identity_5:output:0"P
%lstm_25_while_lstm_25_strided_slice_1'lstm_25_while_lstm_25_strided_slice_1_0"z
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0"|
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0"x
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0"�
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2f
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp2d
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp2h
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_139204
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_27_biasadd_readvariableop_resource:	���)while/lstm_cell_27/BiasAdd/ReadVariableOp�(while/lstm_cell_27/MatMul/ReadVariableOp�*while/lstm_cell_27/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0 while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_3Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_27/Sigmoid_4Sigmoidwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_3:y:0 while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_139061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_27_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_27_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_27_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_27_biasadd_readvariableop_resource:	���)while/lstm_cell_27/BiasAdd/ReadVariableOp�(while/lstm_cell_27/MatMul/ReadVariableOp�*while/lstm_cell_27/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0 while/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_27/Sigmoid_3Sigmoid!while/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_27/Sigmoid_4Sigmoidwhile/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_3:y:0 while/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_134316
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_25_134340_0:	�.
while_lstm_cell_25_134342_0:	Z�*
while_lstm_cell_25_134344_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_25_134340:	�,
while_lstm_cell_25_134342:	Z�(
while_lstm_cell_25_134344:	���*while/lstm_cell_25/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_25_134340_0while_lstm_cell_25_134342_0while_lstm_cell_25_134344_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134302�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������Z�
while/Identity_5Identity3while/lstm_cell_25/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������Zy

while/NoOpNoOp+^while/lstm_cell_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_25_134340while_lstm_cell_25_134340_0"8
while_lstm_cell_25_134342while_lstm_cell_25_134342_0"8
while_lstm_cell_25_134344while_lstm_cell_25_134344_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2X
*while/lstm_cell_25/StatefulPartitionedCall*while/lstm_cell_25/StatefulPartitionedCall: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
��
�
!__inference__wrapped_model_134235
lstm_25_inputS
@sequential_7_lstm_25_lstm_cell_25_matmul_readvariableop_resource:	�U
Bsequential_7_lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:	Z�P
Asequential_7_lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	�S
@sequential_7_lstm_26_lstm_cell_26_matmul_readvariableop_resource:	Z�U
Bsequential_7_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:	Z�P
Asequential_7_lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	�S
@sequential_7_lstm_27_lstm_cell_27_matmul_readvariableop_resource:	Z�U
Bsequential_7_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:	Z�P
Asequential_7_lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	�E
3sequential_7_dense_7_matmul_readvariableop_resource:ZB
4sequential_7_dense_7_biasadd_readvariableop_resource:
identity��+sequential_7/dense_7/BiasAdd/ReadVariableOp�*sequential_7/dense_7/MatMul/ReadVariableOp�8sequential_7/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp�7sequential_7/lstm_25/lstm_cell_25/MatMul/ReadVariableOp�9sequential_7/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp�sequential_7/lstm_25/while�8sequential_7/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp�7sequential_7/lstm_26/lstm_cell_26/MatMul/ReadVariableOp�9sequential_7/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp�sequential_7/lstm_26/while�8sequential_7/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp�7sequential_7/lstm_27/lstm_cell_27/MatMul/ReadVariableOp�9sequential_7/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp�sequential_7/lstm_27/whileW
sequential_7/lstm_25/ShapeShapelstm_25_input*
T0*
_output_shapes
:r
(sequential_7/lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"sequential_7/lstm_25/strided_sliceStridedSlice#sequential_7/lstm_25/Shape:output:01sequential_7/lstm_25/strided_slice/stack:output:03sequential_7/lstm_25/strided_slice/stack_1:output:03sequential_7/lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
!sequential_7/lstm_25/zeros/packedPack+sequential_7/lstm_25/strided_slice:output:0,sequential_7/lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_7/lstm_25/zerosFill*sequential_7/lstm_25/zeros/packed:output:0)sequential_7/lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:���������Zg
%sequential_7/lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
#sequential_7/lstm_25/zeros_1/packedPack+sequential_7/lstm_25/strided_slice:output:0.sequential_7/lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_7/lstm_25/zeros_1Fill,sequential_7/lstm_25/zeros_1/packed:output:0+sequential_7/lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zx
#sequential_7/lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_7/lstm_25/transpose	Transposelstm_25_input,sequential_7/lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:���������n
sequential_7/lstm_25/Shape_1Shape"sequential_7/lstm_25/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_25/strided_slice_1StridedSlice%sequential_7/lstm_25/Shape_1:output:03sequential_7/lstm_25/strided_slice_1/stack:output:05sequential_7/lstm_25/strided_slice_1/stack_1:output:05sequential_7/lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
"sequential_7/lstm_25/TensorArrayV2TensorListReserve9sequential_7/lstm_25/TensorArrayV2/element_shape:output:0-sequential_7/lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Jsequential_7/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
<sequential_7/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_25/transpose:y:0Ssequential_7/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���t
*sequential_7/lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_25/strided_slice_2StridedSlice"sequential_7/lstm_25/transpose:y:03sequential_7/lstm_25/strided_slice_2/stack:output:05sequential_7/lstm_25/strided_slice_2/stack_1:output:05sequential_7/lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
7sequential_7/lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_25_lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
(sequential_7/lstm_25/lstm_cell_25/MatMulMatMul-sequential_7/lstm_25/strided_slice_2:output:0?sequential_7/lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9sequential_7/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_25_lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
*sequential_7/lstm_25/lstm_cell_25/MatMul_1MatMul#sequential_7/lstm_25/zeros:output:0Asequential_7/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%sequential_7/lstm_25/lstm_cell_25/addAddV22sequential_7/lstm_25/lstm_cell_25/MatMul:product:04sequential_7/lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
8sequential_7/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)sequential_7/lstm_25/lstm_cell_25/BiasAddBiasAdd)sequential_7/lstm_25/lstm_cell_25/add:z:0@sequential_7/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
1sequential_7/lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
'sequential_7/lstm_25/lstm_cell_25/splitSplit:sequential_7/lstm_25/lstm_cell_25/split/split_dim:output:02sequential_7/lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
)sequential_7/lstm_25/lstm_cell_25/SigmoidSigmoid0sequential_7/lstm_25/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_25/lstm_cell_25/Sigmoid_1Sigmoid0sequential_7/lstm_25/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
%sequential_7/lstm_25/lstm_cell_25/mulMul/sequential_7/lstm_25/lstm_cell_25/Sigmoid_1:y:0%sequential_7/lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_25/lstm_cell_25/Sigmoid_2Sigmoid0sequential_7/lstm_25/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_25/lstm_cell_25/mul_1Mul-sequential_7/lstm_25/lstm_cell_25/Sigmoid:y:0/sequential_7/lstm_25/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_25/lstm_cell_25/add_1AddV2)sequential_7/lstm_25/lstm_cell_25/mul:z:0+sequential_7/lstm_25/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_25/lstm_cell_25/Sigmoid_3Sigmoid0sequential_7/lstm_25/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_25/lstm_cell_25/Sigmoid_4Sigmoid+sequential_7/lstm_25/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_25/lstm_cell_25/mul_2Mul/sequential_7/lstm_25/lstm_cell_25/Sigmoid_3:y:0/sequential_7/lstm_25/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2sequential_7/lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
$sequential_7/lstm_25/TensorArrayV2_1TensorListReserve;sequential_7/lstm_25/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���[
sequential_7/lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
'sequential_7/lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_7/lstm_25/whileWhile0sequential_7/lstm_25/while/loop_counter:output:06sequential_7/lstm_25/while/maximum_iterations:output:0"sequential_7/lstm_25/time:output:0-sequential_7/lstm_25/TensorArrayV2_1:handle:0#sequential_7/lstm_25/zeros:output:0%sequential_7/lstm_25/zeros_1:output:0-sequential_7/lstm_25/strided_slice_1:output:0Lsequential_7/lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_25_lstm_cell_25_matmul_readvariableop_resourceBsequential_7_lstm_25_lstm_cell_25_matmul_1_readvariableop_resourceAsequential_7_lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_7_lstm_25_while_body_133864*2
cond*R(
&sequential_7_lstm_25_while_cond_133863*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
Esequential_7/lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
7sequential_7/lstm_25/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_25/while:output:3Nsequential_7/lstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0}
*sequential_7/lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������v
,sequential_7/lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_25/strided_slice_3StridedSlice@sequential_7/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_25/strided_slice_3/stack:output:05sequential_7/lstm_25/strided_slice_3/stack_1:output:05sequential_7/lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskz
%sequential_7/lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_7/lstm_25/transpose_1	Transpose@sequential_7/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zp
sequential_7/lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
 sequential_7/dropout_25/IdentityIdentity$sequential_7/lstm_25/transpose_1:y:0*
T0*+
_output_shapes
:���������Zs
sequential_7/lstm_26/ShapeShape)sequential_7/dropout_25/Identity:output:0*
T0*
_output_shapes
:r
(sequential_7/lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"sequential_7/lstm_26/strided_sliceStridedSlice#sequential_7/lstm_26/Shape:output:01sequential_7/lstm_26/strided_slice/stack:output:03sequential_7/lstm_26/strided_slice/stack_1:output:03sequential_7/lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
!sequential_7/lstm_26/zeros/packedPack+sequential_7/lstm_26/strided_slice:output:0,sequential_7/lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_7/lstm_26/zerosFill*sequential_7/lstm_26/zeros/packed:output:0)sequential_7/lstm_26/zeros/Const:output:0*
T0*'
_output_shapes
:���������Zg
%sequential_7/lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
#sequential_7/lstm_26/zeros_1/packedPack+sequential_7/lstm_26/strided_slice:output:0.sequential_7/lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_7/lstm_26/zeros_1Fill,sequential_7/lstm_26/zeros_1/packed:output:0+sequential_7/lstm_26/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zx
#sequential_7/lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_7/lstm_26/transpose	Transpose)sequential_7/dropout_25/Identity:output:0,sequential_7/lstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:���������Zn
sequential_7/lstm_26/Shape_1Shape"sequential_7/lstm_26/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_26/strided_slice_1StridedSlice%sequential_7/lstm_26/Shape_1:output:03sequential_7/lstm_26/strided_slice_1/stack:output:05sequential_7/lstm_26/strided_slice_1/stack_1:output:05sequential_7/lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
"sequential_7/lstm_26/TensorArrayV2TensorListReserve9sequential_7/lstm_26/TensorArrayV2/element_shape:output:0-sequential_7/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Jsequential_7/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
<sequential_7/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_26/transpose:y:0Ssequential_7/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���t
*sequential_7/lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_26/strided_slice_2StridedSlice"sequential_7/lstm_26/transpose:y:03sequential_7/lstm_26/strided_slice_2/stack:output:05sequential_7/lstm_26/strided_slice_2/stack_1:output:05sequential_7/lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
7sequential_7/lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
(sequential_7/lstm_26/lstm_cell_26/MatMulMatMul-sequential_7/lstm_26/strided_slice_2:output:0?sequential_7/lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9sequential_7/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
*sequential_7/lstm_26/lstm_cell_26/MatMul_1MatMul#sequential_7/lstm_26/zeros:output:0Asequential_7/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%sequential_7/lstm_26/lstm_cell_26/addAddV22sequential_7/lstm_26/lstm_cell_26/MatMul:product:04sequential_7/lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
8sequential_7/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)sequential_7/lstm_26/lstm_cell_26/BiasAddBiasAdd)sequential_7/lstm_26/lstm_cell_26/add:z:0@sequential_7/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
1sequential_7/lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
'sequential_7/lstm_26/lstm_cell_26/splitSplit:sequential_7/lstm_26/lstm_cell_26/split/split_dim:output:02sequential_7/lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
)sequential_7/lstm_26/lstm_cell_26/SigmoidSigmoid0sequential_7/lstm_26/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_26/lstm_cell_26/Sigmoid_1Sigmoid0sequential_7/lstm_26/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
%sequential_7/lstm_26/lstm_cell_26/mulMul/sequential_7/lstm_26/lstm_cell_26/Sigmoid_1:y:0%sequential_7/lstm_26/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_26/lstm_cell_26/Sigmoid_2Sigmoid0sequential_7/lstm_26/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_26/lstm_cell_26/mul_1Mul-sequential_7/lstm_26/lstm_cell_26/Sigmoid:y:0/sequential_7/lstm_26/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_26/lstm_cell_26/add_1AddV2)sequential_7/lstm_26/lstm_cell_26/mul:z:0+sequential_7/lstm_26/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_26/lstm_cell_26/Sigmoid_3Sigmoid0sequential_7/lstm_26/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_26/lstm_cell_26/Sigmoid_4Sigmoid+sequential_7/lstm_26/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_26/lstm_cell_26/mul_2Mul/sequential_7/lstm_26/lstm_cell_26/Sigmoid_3:y:0/sequential_7/lstm_26/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2sequential_7/lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
$sequential_7/lstm_26/TensorArrayV2_1TensorListReserve;sequential_7/lstm_26/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���[
sequential_7/lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
'sequential_7/lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_7/lstm_26/whileWhile0sequential_7/lstm_26/while/loop_counter:output:06sequential_7/lstm_26/while/maximum_iterations:output:0"sequential_7/lstm_26/time:output:0-sequential_7/lstm_26/TensorArrayV2_1:handle:0#sequential_7/lstm_26/zeros:output:0%sequential_7/lstm_26/zeros_1:output:0-sequential_7/lstm_26/strided_slice_1:output:0Lsequential_7/lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_26_lstm_cell_26_matmul_readvariableop_resourceBsequential_7_lstm_26_lstm_cell_26_matmul_1_readvariableop_resourceAsequential_7_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_7_lstm_26_while_body_134004*2
cond*R(
&sequential_7_lstm_26_while_cond_134003*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
Esequential_7/lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
7sequential_7/lstm_26/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_26/while:output:3Nsequential_7/lstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0}
*sequential_7/lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������v
,sequential_7/lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_26/strided_slice_3StridedSlice@sequential_7/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_26/strided_slice_3/stack:output:05sequential_7/lstm_26/strided_slice_3/stack_1:output:05sequential_7/lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskz
%sequential_7/lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_7/lstm_26/transpose_1	Transpose@sequential_7/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_26/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zp
sequential_7/lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
 sequential_7/dropout_26/IdentityIdentity$sequential_7/lstm_26/transpose_1:y:0*
T0*+
_output_shapes
:���������Zs
sequential_7/lstm_27/ShapeShape)sequential_7/dropout_26/Identity:output:0*
T0*
_output_shapes
:r
(sequential_7/lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"sequential_7/lstm_27/strided_sliceStridedSlice#sequential_7/lstm_27/Shape:output:01sequential_7/lstm_27/strided_slice/stack:output:03sequential_7/lstm_27/strided_slice/stack_1:output:03sequential_7/lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
!sequential_7/lstm_27/zeros/packedPack+sequential_7/lstm_27/strided_slice:output:0,sequential_7/lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_7/lstm_27/zerosFill*sequential_7/lstm_27/zeros/packed:output:0)sequential_7/lstm_27/zeros/Const:output:0*
T0*'
_output_shapes
:���������Zg
%sequential_7/lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
#sequential_7/lstm_27/zeros_1/packedPack+sequential_7/lstm_27/strided_slice:output:0.sequential_7/lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_7/lstm_27/zeros_1Fill,sequential_7/lstm_27/zeros_1/packed:output:0+sequential_7/lstm_27/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zx
#sequential_7/lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_7/lstm_27/transpose	Transpose)sequential_7/dropout_26/Identity:output:0,sequential_7/lstm_27/transpose/perm:output:0*
T0*+
_output_shapes
:���������Zn
sequential_7/lstm_27/Shape_1Shape"sequential_7/lstm_27/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_27/strided_slice_1StridedSlice%sequential_7/lstm_27/Shape_1:output:03sequential_7/lstm_27/strided_slice_1/stack:output:05sequential_7/lstm_27/strided_slice_1/stack_1:output:05sequential_7/lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
"sequential_7/lstm_27/TensorArrayV2TensorListReserve9sequential_7/lstm_27/TensorArrayV2/element_shape:output:0-sequential_7/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Jsequential_7/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
<sequential_7/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_27/transpose:y:0Ssequential_7/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���t
*sequential_7/lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_27/strided_slice_2StridedSlice"sequential_7/lstm_27/transpose:y:03sequential_7/lstm_27/strided_slice_2/stack:output:05sequential_7/lstm_27/strided_slice_2/stack_1:output:05sequential_7/lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
7sequential_7/lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_27_lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
(sequential_7/lstm_27/lstm_cell_27/MatMulMatMul-sequential_7/lstm_27/strided_slice_2:output:0?sequential_7/lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9sequential_7/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
*sequential_7/lstm_27/lstm_cell_27/MatMul_1MatMul#sequential_7/lstm_27/zeros:output:0Asequential_7/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%sequential_7/lstm_27/lstm_cell_27/addAddV22sequential_7/lstm_27/lstm_cell_27/MatMul:product:04sequential_7/lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
8sequential_7/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)sequential_7/lstm_27/lstm_cell_27/BiasAddBiasAdd)sequential_7/lstm_27/lstm_cell_27/add:z:0@sequential_7/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
1sequential_7/lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
'sequential_7/lstm_27/lstm_cell_27/splitSplit:sequential_7/lstm_27/lstm_cell_27/split/split_dim:output:02sequential_7/lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split�
)sequential_7/lstm_27/lstm_cell_27/SigmoidSigmoid0sequential_7/lstm_27/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_27/lstm_cell_27/Sigmoid_1Sigmoid0sequential_7/lstm_27/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
%sequential_7/lstm_27/lstm_cell_27/mulMul/sequential_7/lstm_27/lstm_cell_27/Sigmoid_1:y:0%sequential_7/lstm_27/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_27/lstm_cell_27/Sigmoid_2Sigmoid0sequential_7/lstm_27/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_27/lstm_cell_27/mul_1Mul-sequential_7/lstm_27/lstm_cell_27/Sigmoid:y:0/sequential_7/lstm_27/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_27/lstm_cell_27/add_1AddV2)sequential_7/lstm_27/lstm_cell_27/mul:z:0+sequential_7/lstm_27/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_27/lstm_cell_27/Sigmoid_3Sigmoid0sequential_7/lstm_27/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Z�
+sequential_7/lstm_27/lstm_cell_27/Sigmoid_4Sigmoid+sequential_7/lstm_27/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
'sequential_7/lstm_27/lstm_cell_27/mul_2Mul/sequential_7/lstm_27/lstm_cell_27/Sigmoid_3:y:0/sequential_7/lstm_27/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
2sequential_7/lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
$sequential_7/lstm_27/TensorArrayV2_1TensorListReserve;sequential_7/lstm_27/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���[
sequential_7/lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
'sequential_7/lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_7/lstm_27/whileWhile0sequential_7/lstm_27/while/loop_counter:output:06sequential_7/lstm_27/while/maximum_iterations:output:0"sequential_7/lstm_27/time:output:0-sequential_7/lstm_27/TensorArrayV2_1:handle:0#sequential_7/lstm_27/zeros:output:0%sequential_7/lstm_27/zeros_1:output:0-sequential_7/lstm_27/strided_slice_1:output:0Lsequential_7/lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_27_lstm_cell_27_matmul_readvariableop_resourceBsequential_7_lstm_27_lstm_cell_27_matmul_1_readvariableop_resourceAsequential_7_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_7_lstm_27_while_body_134144*2
cond*R(
&sequential_7_lstm_27_while_cond_134143*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
Esequential_7/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
7sequential_7/lstm_27/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_27/while:output:3Nsequential_7/lstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0}
*sequential_7/lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������v
,sequential_7/lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_7/lstm_27/strided_slice_3StridedSlice@sequential_7/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_27/strided_slice_3/stack:output:05sequential_7/lstm_27/strided_slice_3/stack_1:output:05sequential_7/lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskz
%sequential_7/lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_7/lstm_27/transpose_1	Transpose@sequential_7/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_27/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zp
sequential_7/lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
 sequential_7/dropout_27/IdentityIdentity-sequential_7/lstm_27/strided_slice_3:output:0*
T0*'
_output_shapes
:���������Z�
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0�
sequential_7/dense_7/MatMulMatMul)sequential_7/dropout_27/Identity:output:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%sequential_7/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp9^sequential_7/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp8^sequential_7/lstm_25/lstm_cell_25/MatMul/ReadVariableOp:^sequential_7/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp^sequential_7/lstm_25/while9^sequential_7/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp8^sequential_7/lstm_26/lstm_cell_26/MatMul/ReadVariableOp:^sequential_7/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^sequential_7/lstm_26/while9^sequential_7/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp8^sequential_7/lstm_27/lstm_cell_27/MatMul/ReadVariableOp:^sequential_7/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^sequential_7/lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp2t
8sequential_7/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp8sequential_7/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_25/lstm_cell_25/MatMul/ReadVariableOp7sequential_7/lstm_25/lstm_cell_25/MatMul/ReadVariableOp2v
9sequential_7/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp9sequential_7/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp28
sequential_7/lstm_25/whilesequential_7/lstm_25/while2t
8sequential_7/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp8sequential_7/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_26/lstm_cell_26/MatMul/ReadVariableOp7sequential_7/lstm_26/lstm_cell_26/MatMul/ReadVariableOp2v
9sequential_7/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp9sequential_7/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp28
sequential_7/lstm_26/whilesequential_7/lstm_26/while2t
8sequential_7/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp8sequential_7/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_27/lstm_cell_27/MatMul/ReadVariableOp7sequential_7/lstm_27/lstm_cell_27/MatMul/ReadVariableOp2v
9sequential_7/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp9sequential_7/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp28
sequential_7/lstm_27/whilesequential_7/lstm_27/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_25_input
�
�
(__inference_lstm_25_layer_call_fn_137551
inputs_0
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_134576|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_138274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_138274___redundant_placeholder04
0while_while_cond_138274___redundant_placeholder14
0while_while_cond_138274___redundant_placeholder24
0while_while_cond_138274___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
� 
�
H__inference_sequential_7_layer_call_and_return_conditional_losses_136526
lstm_25_input!
lstm_25_136496:	�!
lstm_25_136498:	Z�
lstm_25_136500:	�!
lstm_26_136504:	Z�!
lstm_26_136506:	Z�
lstm_26_136508:	�!
lstm_27_136512:	Z�!
lstm_27_136514:	Z�
lstm_27_136516:	� 
dense_7_136520:Z
dense_7_136522:
identity��dense_7/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�lstm_26/StatefulPartitionedCall�lstm_27/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCalllstm_25_inputlstm_25_136496lstm_25_136498lstm_25_136500*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_135434�
dropout_25/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_135447�
lstm_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0lstm_26_136504lstm_26_136506lstm_26_136508*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_135591�
dropout_26/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_26_layer_call_and_return_conditional_losses_135604�
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0lstm_27_136512lstm_27_136514lstm_27_136516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135748�
dropout_27/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_135761�
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_7_136520dense_7_136522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_135773w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_7/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_25_input
Ζ
�

H__inference_sequential_7_layer_call_and_return_conditional_losses_137529

inputsF
3lstm_25_lstm_cell_25_matmul_readvariableop_resource:	�H
5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:	Z�C
4lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	�F
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	Z�H
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:	Z�C
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	�F
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:	Z�H
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:	Z�C
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	�8
&dense_7_matmul_readvariableop_resource:Z5
'dense_7_biasadd_readvariableop_resource:
identity��dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp�*lstm_25/lstm_cell_25/MatMul/ReadVariableOp�,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp�lstm_25/while�+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp�*lstm_26/lstm_cell_26/MatMul/ReadVariableOp�,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp�lstm_26/while�+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp�*lstm_27/lstm_cell_27/MatMul/ReadVariableOp�,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp�lstm_27/whileC
lstm_25/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_sliceStridedSlicelstm_25/Shape:output:0$lstm_25/strided_slice/stack:output:0&lstm_25/strided_slice/stack_1:output:0&lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_25/zeros/packedPacklstm_25/strided_slice:output:0lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:���������ZZ
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_25/zeros_1/packedPacklstm_25/strided_slice:output:0!lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zk
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_25/transpose	Transposeinputslstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_25/Shape_1Shapelstm_25/transpose:y:0*
T0*
_output_shapes
:g
lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_slice_1StridedSlicelstm_25/Shape_1:output:0&lstm_25/strided_slice_1/stack:output:0(lstm_25/strided_slice_1/stack_1:output:0(lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_25/TensorArrayV2TensorListReserve,lstm_25/TensorArrayV2/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_25/transpose:y:0Flstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_25/lstm_cell_25/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_25/lstm_cell_25/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_25/lstm_cell_25/addAddV2%lstm_25/lstm_cell_25/MatMul:product:0'lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_25/lstm_cell_25/BiasAddBiasAddlstm_25/lstm_cell_25/add:z:03lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_25/lstm_cell_25/splitSplit-lstm_25/lstm_cell_25/split/split_dim:output:0%lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split~
lstm_25/lstm_cell_25/SigmoidSigmoid#lstm_25/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/Sigmoid_1Sigmoid#lstm_25/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/mulMul"lstm_25/lstm_cell_25/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/Sigmoid_2Sigmoid#lstm_25/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/mul_1Mul lstm_25/lstm_cell_25/Sigmoid:y:0"lstm_25/lstm_cell_25/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/add_1AddV2lstm_25/lstm_cell_25/mul:z:0lstm_25/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/Sigmoid_3Sigmoid#lstm_25/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:���������Z{
lstm_25/lstm_cell_25/Sigmoid_4Sigmoidlstm_25/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_25/lstm_cell_25/mul_2Mul"lstm_25/lstm_cell_25/Sigmoid_3:y:0"lstm_25/lstm_cell_25/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zv
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
lstm_25/TensorArrayV2_1TensorListReserve.lstm_25/TensorArrayV2_1/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_25_matmul_readvariableop_resource5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_25_while_body_137137*%
condR
lstm_25_while_cond_137136*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0p
lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskm
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zc
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?�
dropout_25/dropout/MulMullstm_25/transpose_1:y:0!dropout_25/dropout/Const:output:0*
T0*+
_output_shapes
:���������Z_
dropout_25/dropout/ShapeShapelstm_25/transpose_1:y:0*
T0*
_output_shapes
:�
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*+
_output_shapes
:���������Z*
dtype0f
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z�
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z�
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*+
_output_shapes
:���������ZY
lstm_26/ShapeShapedropout_25/dropout/Mul_1:z:0*
T0*
_output_shapes
:e
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_sliceStridedSlicelstm_26/Shape:output:0$lstm_26/strided_slice/stack:output:0&lstm_26/strided_slice/stack_1:output:0&lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_26/zeros/packedPacklstm_26/strided_slice:output:0lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*'
_output_shapes
:���������ZZ
lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_26/zeros_1/packedPacklstm_26/strided_slice:output:0!lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zk
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_26/transpose	Transposedropout_25/dropout/Mul_1:z:0lstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:���������ZT
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:g
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split~
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0"lstm_26/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/Sigmoid_3Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Z{
lstm_26/lstm_cell_26/Sigmoid_4Sigmoidlstm_26/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_3:y:0"lstm_26/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zv
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
lstm_26/TensorArrayV2_1TensorListReserve.lstm_26/TensorArrayV2_1/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_26_while_body_137284*%
condR
lstm_26_while_cond_137283*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0p
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskm
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zc
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?�
dropout_26/dropout/MulMullstm_26/transpose_1:y:0!dropout_26/dropout/Const:output:0*
T0*+
_output_shapes
:���������Z_
dropout_26/dropout/ShapeShapelstm_26/transpose_1:y:0*
T0*
_output_shapes
:�
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*+
_output_shapes
:���������Z*
dtype0f
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z�
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z�
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*+
_output_shapes
:���������ZY
lstm_27/ShapeShapedropout_26/dropout/Mul_1:z:0*
T0*
_output_shapes
:e
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*'
_output_shapes
:���������ZZ
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z�
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zk
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_27/transpose	Transposedropout_26/dropout/Mul_1:z:0lstm_27/transpose/perm:output:0*
T0*+
_output_shapes
:���������ZT
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:g
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_split~
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0"lstm_27/lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/Sigmoid_3Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Z{
lstm_27/lstm_cell_27/Sigmoid_4Sigmoidlstm_27/lstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_3:y:0"lstm_27/lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zv
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_27_while_body_137431*%
condR
lstm_27_while_cond_137430*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0p
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maskm
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Zc
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *X��?�
dropout_27/dropout/MulMul lstm_27/strided_slice_3:output:0!dropout_27/dropout/Const:output:0*
T0*'
_output_shapes
:���������Zh
dropout_27/dropout/ShapeShape lstm_27/strided_slice_3:output:0*
T0*
_output_shapes
:�
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*'
_output_shapes
:���������Z*
dtype0f
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������Z�
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������Z�
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*'
_output_shapes
:���������Z�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0�
dense_7/MatMulMatMuldropout_27/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp,^lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+^lstm_25/lstm_cell_25/MatMul/ReadVariableOp-^lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp^lstm_25/while,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2Z
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp2X
*lstm_25/lstm_cell_25/MatMul/ReadVariableOp*lstm_25/lstm_cell_25/MatMul/ReadVariableOp2\
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp2
lstm_25/whilelstm_25/while2Z
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2X
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp*lstm_26/lstm_cell_26/MatMul/ReadVariableOp2\
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2
lstm_26/whilelstm_26/while2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
H__inference_sequential_7_layer_call_and_return_conditional_losses_136559
lstm_25_input!
lstm_25_136529:	�!
lstm_25_136531:	Z�
lstm_25_136533:	�!
lstm_26_136537:	Z�!
lstm_26_136539:	Z�
lstm_26_136541:	�!
lstm_27_136545:	Z�!
lstm_27_136547:	Z�
lstm_27_136549:	� 
dense_7_136553:Z
dense_7_136555:
identity��dense_7/StatefulPartitionedCall�"dropout_25/StatefulPartitionedCall�"dropout_26/StatefulPartitionedCall�"dropout_27/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�lstm_26/StatefulPartitionedCall�lstm_27/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCalllstm_25_inputlstm_25_136529lstm_25_136531lstm_25_136533*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_136370�
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_136211�
lstm_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0lstm_26_136537lstm_26_136539lstm_26_136541*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_26_layer_call_and_return_conditional_losses_136182�
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_26_layer_call_and_return_conditional_losses_136023�
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0lstm_27_136545lstm_27_136547lstm_27_136549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_27_layer_call_and_return_conditional_losses_135994�
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_26/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_135835�
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_7_136553dense_7_136555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_135773w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_25_input
�
�
(__inference_lstm_25_layer_call_fn_137573

inputs
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_25_layer_call_and_return_conditional_losses_136370s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
+__inference_dropout_25_layer_call_fn_138155

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_25_layer_call_and_return_conditional_losses_136211s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�J
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_135994

inputs>
+lstm_cell_27_matmul_readvariableop_resource:	Z�@
-lstm_cell_27_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_27_biasadd_readvariableop_resource:	�
identity��#lstm_cell_27/BiasAdd/ReadVariableOp�"lstm_cell_27/MatMul/ReadVariableOp�$lstm_cell_27/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_3Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_27/Sigmoid_4Sigmoidlstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_3:y:0lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_135910*
condR
while_cond_135909*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������Z: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�8
�
while_body_138418
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	Z�H
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:	Z�C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	Z�F
3while_lstm_cell_26_matmul_1_readvariableop_resource:	Z�A
2while_lstm_cell_26_biasadd_readvariableop_resource:	���)while/lstm_cell_26/BiasAdd/ReadVariableOp�(while/lstm_cell_26/MatMul/ReadVariableOp�*while/lstm_cell_26/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������Z*
element_dtype0�
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	Z�*
dtype0�
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitz
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:2*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0 while/lstm_cell_26/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*'
_output_shapes
:���������Z|
while/lstm_cell_26/Sigmoid_3Sigmoid!while/lstm_cell_26/split:output:3*
T0*'
_output_shapes
:���������Zw
while/lstm_cell_26/Sigmoid_4Sigmoidwhile/lstm_cell_26/add_1:z:0*
T0*'
_output_shapes
:���������Z�
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_3:y:0 while/lstm_cell_26/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Z�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������Zy
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������Z�

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_134507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_25_134531_0:	�.
while_lstm_cell_25_134533_0:	Z�*
while_lstm_cell_25_134535_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_25_134531:	�,
while_lstm_cell_25_134533:	Z�(
while_lstm_cell_25_134535:	���*while/lstm_cell_25/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_25_134531_0while_lstm_cell_25_134533_0while_lstm_cell_25_134535_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134448�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������Z�
while/Identity_5Identity3while/lstm_cell_25/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������Zy

while/NoOpNoOp+^while/lstm_cell_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_25_134531while_lstm_cell_25_134531_0"8
while_lstm_cell_25_134533while_lstm_cell_25_134533_0"8
while_lstm_cell_25_134535while_lstm_cell_25_134535_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������Z:���������Z: : : : : 2X
*while/lstm_cell_25/StatefulPartitionedCall*while/lstm_cell_25/StatefulPartitionedCall: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_25_layer_call_fn_139494

inputs
states_0
states_1
unknown:	�
	unknown_0:	Z�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������Z:���������Z:���������Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_134302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������Z:���������Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�
�
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_135002

inputs

states
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates:OK
'
_output_shapes
:���������Z
 
_user_specified_namestates
�
�
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_139641

inputs
states_0
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1
�
�
while_cond_137774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_137774___redundant_placeholder04
0while_while_cond_137774___redundant_placeholder14
0while_while_cond_137774___redundant_placeholder24
0while_while_cond_137774___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
d
+__inference_dropout_26_layer_call_fn_138798

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_26_layer_call_and_return_conditional_losses_136023s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
while_cond_134856
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_134856___redundant_placeholder04
0while_while_cond_134856___redundant_placeholder14
0while_while_cond_134856___redundant_placeholder24
0while_while_cond_134856___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�J
�
C__inference_lstm_27_layer_call_and_return_conditional_losses_139002
inputs_0>
+lstm_cell_27_matmul_readvariableop_resource:	Z�@
-lstm_cell_27_matmul_1_readvariableop_resource:	Z�;
,lstm_cell_27_biasadd_readvariableop_resource:	�
identity��#lstm_cell_27/BiasAdd/ReadVariableOp�"lstm_cell_27/MatMul/ReadVariableOp�$lstm_cell_27/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_mask�
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0�
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitn
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*'
_output_shapes
:���������Zw
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:2*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Sigmoid_2:y:0*
T0*'
_output_shapes
:���������Z{
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*'
_output_shapes
:���������Zp
lstm_cell_27/Sigmoid_3Sigmoidlstm_cell_27/split:output:3*
T0*'
_output_shapes
:���������Zk
lstm_cell_27/Sigmoid_4Sigmoidlstm_cell_27/add_1:z:0*
T0*'
_output_shapes
:���������Z�
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_3:y:0lstm_cell_27/Sigmoid_4:y:0*
T0*'
_output_shapes
:���������Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������Z:���������Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_138918*
condR
while_cond_138917*K
output_shapes:
8: : : : :���������Z:���������Z: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������Z: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������Z
"
_user_specified_name
inputs/0
�
�
while_cond_135349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_135349___redundant_placeholder04
0while_while_cond_135349___redundant_placeholder14
0while_while_cond_135349___redundant_placeholder24
0while_while_cond_135349___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_134665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_134665___redundant_placeholder04
0while_while_cond_134665___redundant_placeholder14
0while_while_cond_134665___redundant_placeholder24
0while_while_cond_134665___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������Z:���������Z: ::::: 
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
:���������Z:-)
'
_output_shapes
:���������Z:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_139739

inputs
states_0
states_11
matmul_readvariableop_resource:	Z�3
 matmul_1_readvariableop_resource:	Z�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Z�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	Z�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������Z:���������Z:���������Z:���������Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������ZV
	Sigmoid_2Sigmoidsplit:output:2*
T0*'
_output_shapes
:���������ZZ
mul_1MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:���������ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������ZV
	Sigmoid_3Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������ZQ
	Sigmoid_4Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������Z\
mul_2MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������Z:���������Z:���������Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������Z
"
_user_specified_name
states/1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_25_input:
serving_default_lstm_25_input:0���������;
dense_70
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_sequential
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"cell
#
state_spec
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

2beta_1

3beta_2
	4decay
5learning_rate
6iter,m�-m�7m�8m�9m�:m�;m�<m�=m�>m�?m�,v�-v�7v�8v�9v�:v�;v�<v�=v�>v�?v�"
	optimizer
n
70
81
92
:3
;4
<5
=6
>7
?8
,9
-10"
trackable_list_wrapper
n
70
81
92
:3
;4
<5
=6
>7
?8
,9
-10"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
		variables

trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
E
state_size

7kernel
8recurrent_kernel
9bias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
U
state_size

:kernel
;recurrent_kernel
<bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Zstates
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
 regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
e
state_size

=kernel
>recurrent_kernel
?bias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

jstates
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
$	variables
%trainable_variables
&regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :Z2dense_7/kernel
:2dense_7/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
.	variables
/trainable_variables
0regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
.:,	�2lstm_25/lstm_cell_25/kernel
8:6	Z�2%lstm_25/lstm_cell_25/recurrent_kernel
(:&�2lstm_25/lstm_cell_25/bias
.:,	Z�2lstm_26/lstm_cell_26/kernel
8:6	Z�2%lstm_26/lstm_cell_26/recurrent_kernel
(:&�2lstm_26/lstm_cell_26/bias
.:,	Z�2lstm_27/lstm_cell_27/kernel
8:6	Z�2%lstm_27/lstm_cell_27/recurrent_kernel
(:&�2lstm_27/lstm_cell_27/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
5
z0
{1
|2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
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
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
%:#Z2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
3:1	�2"Adam/lstm_25/lstm_cell_25/kernel/m
=:;	Z�2,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m
-:+�2 Adam/lstm_25/lstm_cell_25/bias/m
3:1	Z�2"Adam/lstm_26/lstm_cell_26/kernel/m
=:;	Z�2,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m
-:+�2 Adam/lstm_26/lstm_cell_26/bias/m
3:1	Z�2"Adam/lstm_27/lstm_cell_27/kernel/m
=:;	Z�2,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m
-:+�2 Adam/lstm_27/lstm_cell_27/bias/m
%:#Z2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
3:1	�2"Adam/lstm_25/lstm_cell_25/kernel/v
=:;	Z�2,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v
-:+�2 Adam/lstm_25/lstm_cell_25/bias/v
3:1	Z�2"Adam/lstm_26/lstm_cell_26/kernel/v
=:;	Z�2,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v
-:+�2 Adam/lstm_26/lstm_cell_26/bias/v
3:1	Z�2"Adam/lstm_27/lstm_cell_27/kernel/v
=:;	Z�2,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v
-:+�2 Adam/lstm_27/lstm_cell_27/bias/v
�2�
-__inference_sequential_7_layer_call_fn_135805
-__inference_sequential_7_layer_call_fn_136621
-__inference_sequential_7_layer_call_fn_136648
-__inference_sequential_7_layer_call_fn_136493�
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_137078
H__inference_sequential_7_layer_call_and_return_conditional_losses_137529
H__inference_sequential_7_layer_call_and_return_conditional_losses_136526
H__inference_sequential_7_layer_call_and_return_conditional_losses_136559�
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
�B�
!__inference__wrapped_model_134235lstm_25_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_lstm_25_layer_call_fn_137540
(__inference_lstm_25_layer_call_fn_137551
(__inference_lstm_25_layer_call_fn_137562
(__inference_lstm_25_layer_call_fn_137573�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_25_layer_call_and_return_conditional_losses_137716
C__inference_lstm_25_layer_call_and_return_conditional_losses_137859
C__inference_lstm_25_layer_call_and_return_conditional_losses_138002
C__inference_lstm_25_layer_call_and_return_conditional_losses_138145�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_25_layer_call_fn_138150
+__inference_dropout_25_layer_call_fn_138155�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_25_layer_call_and_return_conditional_losses_138160
F__inference_dropout_25_layer_call_and_return_conditional_losses_138172�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_lstm_26_layer_call_fn_138183
(__inference_lstm_26_layer_call_fn_138194
(__inference_lstm_26_layer_call_fn_138205
(__inference_lstm_26_layer_call_fn_138216�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_26_layer_call_and_return_conditional_losses_138359
C__inference_lstm_26_layer_call_and_return_conditional_losses_138502
C__inference_lstm_26_layer_call_and_return_conditional_losses_138645
C__inference_lstm_26_layer_call_and_return_conditional_losses_138788�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_26_layer_call_fn_138793
+__inference_dropout_26_layer_call_fn_138798�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_26_layer_call_and_return_conditional_losses_138803
F__inference_dropout_26_layer_call_and_return_conditional_losses_138815�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_lstm_27_layer_call_fn_138826
(__inference_lstm_27_layer_call_fn_138837
(__inference_lstm_27_layer_call_fn_138848
(__inference_lstm_27_layer_call_fn_138859�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_27_layer_call_and_return_conditional_losses_139002
C__inference_lstm_27_layer_call_and_return_conditional_losses_139145
C__inference_lstm_27_layer_call_and_return_conditional_losses_139288
C__inference_lstm_27_layer_call_and_return_conditional_losses_139431�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_27_layer_call_fn_139436
+__inference_dropout_27_layer_call_fn_139441�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_27_layer_call_and_return_conditional_losses_139446
F__inference_dropout_27_layer_call_and_return_conditional_losses_139458�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dense_7_layer_call_fn_139467�
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
C__inference_dense_7_layer_call_and_return_conditional_losses_139477�
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
$__inference_signature_wrapper_136594lstm_25_input"�
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
 
�2�
-__inference_lstm_cell_25_layer_call_fn_139494
-__inference_lstm_cell_25_layer_call_fn_139511�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_139543
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_139575�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_lstm_cell_26_layer_call_fn_139592
-__inference_lstm_cell_26_layer_call_fn_139609�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_139641
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_139673�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_lstm_cell_27_layer_call_fn_139690
-__inference_lstm_cell_27_layer_call_fn_139707�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_139739
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_139771�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
!__inference__wrapped_model_134235|789:;<=>?,-:�7
0�-
+�(
lstm_25_input���������
� "1�.
,
dense_7!�
dense_7����������
C__inference_dense_7_layer_call_and_return_conditional_losses_139477\,-/�,
%�"
 �
inputs���������Z
� "%�"
�
0���������
� {
(__inference_dense_7_layer_call_fn_139467O,-/�,
%�"
 �
inputs���������Z
� "�����������
F__inference_dropout_25_layer_call_and_return_conditional_losses_138160d7�4
-�*
$�!
inputs���������Z
p 
� ")�&
�
0���������Z
� �
F__inference_dropout_25_layer_call_and_return_conditional_losses_138172d7�4
-�*
$�!
inputs���������Z
p
� ")�&
�
0���������Z
� �
+__inference_dropout_25_layer_call_fn_138150W7�4
-�*
$�!
inputs���������Z
p 
� "����������Z�
+__inference_dropout_25_layer_call_fn_138155W7�4
-�*
$�!
inputs���������Z
p
� "����������Z�
F__inference_dropout_26_layer_call_and_return_conditional_losses_138803d7�4
-�*
$�!
inputs���������Z
p 
� ")�&
�
0���������Z
� �
F__inference_dropout_26_layer_call_and_return_conditional_losses_138815d7�4
-�*
$�!
inputs���������Z
p
� ")�&
�
0���������Z
� �
+__inference_dropout_26_layer_call_fn_138793W7�4
-�*
$�!
inputs���������Z
p 
� "����������Z�
+__inference_dropout_26_layer_call_fn_138798W7�4
-�*
$�!
inputs���������Z
p
� "����������Z�
F__inference_dropout_27_layer_call_and_return_conditional_losses_139446\3�0
)�&
 �
inputs���������Z
p 
� "%�"
�
0���������Z
� �
F__inference_dropout_27_layer_call_and_return_conditional_losses_139458\3�0
)�&
 �
inputs���������Z
p
� "%�"
�
0���������Z
� ~
+__inference_dropout_27_layer_call_fn_139436O3�0
)�&
 �
inputs���������Z
p 
� "����������Z~
+__inference_dropout_27_layer_call_fn_139441O3�0
)�&
 �
inputs���������Z
p
� "����������Z�
C__inference_lstm_25_layer_call_and_return_conditional_losses_137716�789O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "2�/
(�%
0������������������Z
� �
C__inference_lstm_25_layer_call_and_return_conditional_losses_137859�789O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "2�/
(�%
0������������������Z
� �
C__inference_lstm_25_layer_call_and_return_conditional_losses_138002q789?�<
5�2
$�!
inputs���������

 
p 

 
� ")�&
�
0���������Z
� �
C__inference_lstm_25_layer_call_and_return_conditional_losses_138145q789?�<
5�2
$�!
inputs���������

 
p

 
� ")�&
�
0���������Z
� �
(__inference_lstm_25_layer_call_fn_137540}789O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"������������������Z�
(__inference_lstm_25_layer_call_fn_137551}789O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"������������������Z�
(__inference_lstm_25_layer_call_fn_137562d789?�<
5�2
$�!
inputs���������

 
p 

 
� "����������Z�
(__inference_lstm_25_layer_call_fn_137573d789?�<
5�2
$�!
inputs���������

 
p

 
� "����������Z�
C__inference_lstm_26_layer_call_and_return_conditional_losses_138359�:;<O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p 

 
� "2�/
(�%
0������������������Z
� �
C__inference_lstm_26_layer_call_and_return_conditional_losses_138502�:;<O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p

 
� "2�/
(�%
0������������������Z
� �
C__inference_lstm_26_layer_call_and_return_conditional_losses_138645q:;<?�<
5�2
$�!
inputs���������Z

 
p 

 
� ")�&
�
0���������Z
� �
C__inference_lstm_26_layer_call_and_return_conditional_losses_138788q:;<?�<
5�2
$�!
inputs���������Z

 
p

 
� ")�&
�
0���������Z
� �
(__inference_lstm_26_layer_call_fn_138183}:;<O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p 

 
� "%�"������������������Z�
(__inference_lstm_26_layer_call_fn_138194}:;<O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p

 
� "%�"������������������Z�
(__inference_lstm_26_layer_call_fn_138205d:;<?�<
5�2
$�!
inputs���������Z

 
p 

 
� "����������Z�
(__inference_lstm_26_layer_call_fn_138216d:;<?�<
5�2
$�!
inputs���������Z

 
p

 
� "����������Z�
C__inference_lstm_27_layer_call_and_return_conditional_losses_139002}=>?O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p 

 
� "%�"
�
0���������Z
� �
C__inference_lstm_27_layer_call_and_return_conditional_losses_139145}=>?O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p

 
� "%�"
�
0���������Z
� �
C__inference_lstm_27_layer_call_and_return_conditional_losses_139288m=>??�<
5�2
$�!
inputs���������Z

 
p 

 
� "%�"
�
0���������Z
� �
C__inference_lstm_27_layer_call_and_return_conditional_losses_139431m=>??�<
5�2
$�!
inputs���������Z

 
p

 
� "%�"
�
0���������Z
� �
(__inference_lstm_27_layer_call_fn_138826p=>?O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p 

 
� "����������Z�
(__inference_lstm_27_layer_call_fn_138837p=>?O�L
E�B
4�1
/�,
inputs/0������������������Z

 
p

 
� "����������Z�
(__inference_lstm_27_layer_call_fn_138848`=>??�<
5�2
$�!
inputs���������Z

 
p 

 
� "����������Z�
(__inference_lstm_27_layer_call_fn_138859`=>??�<
5�2
$�!
inputs���������Z

 
p

 
� "����������Z�
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_139543�789��}
v�s
 �
inputs���������
K�H
"�
states/0���������Z
"�
states/1���������Z
p 
� "s�p
i�f
�
0/0���������Z
E�B
�
0/1/0���������Z
�
0/1/1���������Z
� �
H__inference_lstm_cell_25_layer_call_and_return_conditional_losses_139575�789��}
v�s
 �
inputs���������
K�H
"�
states/0���������Z
"�
states/1���������Z
p
� "s�p
i�f
�
0/0���������Z
E�B
�
0/1/0���������Z
�
0/1/1���������Z
� �
-__inference_lstm_cell_25_layer_call_fn_139494�789��}
v�s
 �
inputs���������
K�H
"�
states/0���������Z
"�
states/1���������Z
p 
� "c�`
�
0���������Z
A�>
�
1/0���������Z
�
1/1���������Z�
-__inference_lstm_cell_25_layer_call_fn_139511�789��}
v�s
 �
inputs���������
K�H
"�
states/0���������Z
"�
states/1���������Z
p
� "c�`
�
0���������Z
A�>
�
1/0���������Z
�
1/1���������Z�
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_139641�:;<��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p 
� "s�p
i�f
�
0/0���������Z
E�B
�
0/1/0���������Z
�
0/1/1���������Z
� �
H__inference_lstm_cell_26_layer_call_and_return_conditional_losses_139673�:;<��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p
� "s�p
i�f
�
0/0���������Z
E�B
�
0/1/0���������Z
�
0/1/1���������Z
� �
-__inference_lstm_cell_26_layer_call_fn_139592�:;<��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p 
� "c�`
�
0���������Z
A�>
�
1/0���������Z
�
1/1���������Z�
-__inference_lstm_cell_26_layer_call_fn_139609�:;<��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p
� "c�`
�
0���������Z
A�>
�
1/0���������Z
�
1/1���������Z�
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_139739�=>?��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p 
� "s�p
i�f
�
0/0���������Z
E�B
�
0/1/0���������Z
�
0/1/1���������Z
� �
H__inference_lstm_cell_27_layer_call_and_return_conditional_losses_139771�=>?��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p
� "s�p
i�f
�
0/0���������Z
E�B
�
0/1/0���������Z
�
0/1/1���������Z
� �
-__inference_lstm_cell_27_layer_call_fn_139690�=>?��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p 
� "c�`
�
0���������Z
A�>
�
1/0���������Z
�
1/1���������Z�
-__inference_lstm_cell_27_layer_call_fn_139707�=>?��}
v�s
 �
inputs���������Z
K�H
"�
states/0���������Z
"�
states/1���������Z
p
� "c�`
�
0���������Z
A�>
�
1/0���������Z
�
1/1���������Z�
H__inference_sequential_7_layer_call_and_return_conditional_losses_136526x789:;<=>?,-B�?
8�5
+�(
lstm_25_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_7_layer_call_and_return_conditional_losses_136559x789:;<=>?,-B�?
8�5
+�(
lstm_25_input���������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_7_layer_call_and_return_conditional_losses_137078q789:;<=>?,-;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_7_layer_call_and_return_conditional_losses_137529q789:;<=>?,-;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_7_layer_call_fn_135805k789:;<=>?,-B�?
8�5
+�(
lstm_25_input���������
p 

 
� "�����������
-__inference_sequential_7_layer_call_fn_136493k789:;<=>?,-B�?
8�5
+�(
lstm_25_input���������
p

 
� "�����������
-__inference_sequential_7_layer_call_fn_136621d789:;<=>?,-;�8
1�.
$�!
inputs���������
p 

 
� "�����������
-__inference_sequential_7_layer_call_fn_136648d789:;<=>?,-;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_136594�789:;<=>?,-K�H
� 
A�>
<
lstm_25_input+�(
lstm_25_input���������"1�.
,
dense_7!�
dense_7���������