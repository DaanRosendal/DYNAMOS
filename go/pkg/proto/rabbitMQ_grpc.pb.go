// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.4.0
// - protoc             v3.12.4
// source: rabbitMQ.proto

// package proto;

package proto

import (
	context "context"
	empty "github.com/golang/protobuf/ptypes/empty"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.62.0 or later.
const _ = grpc.SupportPackageIsVersion8

const (
	SideCar_InitRabbitMq_FullMethodName                = "/dynamos.SideCar/InitRabbitMq"
	SideCar_InitRabbitForChain_FullMethodName          = "/dynamos.SideCar/InitRabbitForChain"
	SideCar_StopReceivingRabbit_FullMethodName         = "/dynamos.SideCar/StopReceivingRabbit"
	SideCar_Consume_FullMethodName                     = "/dynamos.SideCar/Consume"
	SideCar_ChainConsume_FullMethodName                = "/dynamos.SideCar/ChainConsume"
	SideCar_SendRequestApproval_FullMethodName         = "/dynamos.SideCar/SendRequestApproval"
	SideCar_SendValidationResponse_FullMethodName      = "/dynamos.SideCar/SendValidationResponse"
	SideCar_SendCompositionRequest_FullMethodName      = "/dynamos.SideCar/SendCompositionRequest"
	SideCar_SendSqlDataRequest_FullMethodName          = "/dynamos.SideCar/SendSqlDataRequest"
	SideCar_SendPolicyUpdate_FullMethodName            = "/dynamos.SideCar/SendPolicyUpdate"
	SideCar_SendTest_FullMethodName                    = "/dynamos.SideCar/SendTest"
	SideCar_SendMicroserviceComm_FullMethodName        = "/dynamos.SideCar/SendMicroserviceComm"
	SideCar_CreateQueue_FullMethodName                 = "/dynamos.SideCar/CreateQueue"
	SideCar_DeleteQueue_FullMethodName                 = "/dynamos.SideCar/DeleteQueue"
	SideCar_SendRequestApprovalResponse_FullMethodName = "/dynamos.SideCar/SendRequestApprovalResponse"
	SideCar_SendRequestApprovalRequest_FullMethodName  = "/dynamos.SideCar/SendRequestApprovalRequest"
)

// SideCarClient is the client API for SideCar service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// The sidecar definition.
type SideCarClient interface {
	InitRabbitMq(ctx context.Context, in *InitRequest, opts ...grpc.CallOption) (*empty.Empty, error)
	InitRabbitForChain(ctx context.Context, in *ChainRequest, opts ...grpc.CallOption) (*empty.Empty, error)
	StopReceivingRabbit(ctx context.Context, in *StopRequest, opts ...grpc.CallOption) (*empty.Empty, error)
	Consume(ctx context.Context, in *ConsumeRequest, opts ...grpc.CallOption) (SideCar_ConsumeClient, error)
	ChainConsume(ctx context.Context, in *ConsumeRequest, opts ...grpc.CallOption) (SideCar_ChainConsumeClient, error)
	SendRequestApproval(ctx context.Context, in *RequestApproval, opts ...grpc.CallOption) (*empty.Empty, error)
	SendValidationResponse(ctx context.Context, in *ValidationResponse, opts ...grpc.CallOption) (*empty.Empty, error)
	SendCompositionRequest(ctx context.Context, in *CompositionRequest, opts ...grpc.CallOption) (*empty.Empty, error)
	SendSqlDataRequest(ctx context.Context, in *SqlDataRequest, opts ...grpc.CallOption) (*empty.Empty, error)
	SendPolicyUpdate(ctx context.Context, in *PolicyUpdate, opts ...grpc.CallOption) (*empty.Empty, error)
	SendTest(ctx context.Context, in *SqlDataRequest, opts ...grpc.CallOption) (*empty.Empty, error)
	SendMicroserviceComm(ctx context.Context, in *MicroserviceCommunication, opts ...grpc.CallOption) (*empty.Empty, error)
	CreateQueue(ctx context.Context, in *QueueInfo, opts ...grpc.CallOption) (*empty.Empty, error)
	DeleteQueue(ctx context.Context, in *QueueInfo, opts ...grpc.CallOption) (*empty.Empty, error)
	SendRequestApprovalResponse(ctx context.Context, in *RequestApprovalResponse, opts ...grpc.CallOption) (*empty.Empty, error)
	SendRequestApprovalRequest(ctx context.Context, in *RequestApproval, opts ...grpc.CallOption) (*empty.Empty, error)
}

type sideCarClient struct {
	cc grpc.ClientConnInterface
}

func NewSideCarClient(cc grpc.ClientConnInterface) SideCarClient {
	return &sideCarClient{cc}
}

func (c *sideCarClient) InitRabbitMq(ctx context.Context, in *InitRequest, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_InitRabbitMq_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) InitRabbitForChain(ctx context.Context, in *ChainRequest, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_InitRabbitForChain_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) StopReceivingRabbit(ctx context.Context, in *StopRequest, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_StopReceivingRabbit_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) Consume(ctx context.Context, in *ConsumeRequest, opts ...grpc.CallOption) (SideCar_ConsumeClient, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &SideCar_ServiceDesc.Streams[0], SideCar_Consume_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &sideCarConsumeClient{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type SideCar_ConsumeClient interface {
	Recv() (*SideCarMessage, error)
	grpc.ClientStream
}

type sideCarConsumeClient struct {
	grpc.ClientStream
}

func (x *sideCarConsumeClient) Recv() (*SideCarMessage, error) {
	m := new(SideCarMessage)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *sideCarClient) ChainConsume(ctx context.Context, in *ConsumeRequest, opts ...grpc.CallOption) (SideCar_ChainConsumeClient, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &SideCar_ServiceDesc.Streams[1], SideCar_ChainConsume_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &sideCarChainConsumeClient{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type SideCar_ChainConsumeClient interface {
	Recv() (*SideCarMessage, error)
	grpc.ClientStream
}

type sideCarChainConsumeClient struct {
	grpc.ClientStream
}

func (x *sideCarChainConsumeClient) Recv() (*SideCarMessage, error) {
	m := new(SideCarMessage)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *sideCarClient) SendRequestApproval(ctx context.Context, in *RequestApproval, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendRequestApproval_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendValidationResponse(ctx context.Context, in *ValidationResponse, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendValidationResponse_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendCompositionRequest(ctx context.Context, in *CompositionRequest, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendCompositionRequest_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendSqlDataRequest(ctx context.Context, in *SqlDataRequest, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendSqlDataRequest_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendPolicyUpdate(ctx context.Context, in *PolicyUpdate, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendPolicyUpdate_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendTest(ctx context.Context, in *SqlDataRequest, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendTest_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendMicroserviceComm(ctx context.Context, in *MicroserviceCommunication, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendMicroserviceComm_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) CreateQueue(ctx context.Context, in *QueueInfo, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_CreateQueue_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) DeleteQueue(ctx context.Context, in *QueueInfo, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_DeleteQueue_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendRequestApprovalResponse(ctx context.Context, in *RequestApprovalResponse, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendRequestApprovalResponse_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *sideCarClient) SendRequestApprovalRequest(ctx context.Context, in *RequestApproval, opts ...grpc.CallOption) (*empty.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(empty.Empty)
	err := c.cc.Invoke(ctx, SideCar_SendRequestApprovalRequest_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// SideCarServer is the server API for SideCar service.
// All implementations must embed UnimplementedSideCarServer
// for forward compatibility
//
// The sidecar definition.
type SideCarServer interface {
	InitRabbitMq(context.Context, *InitRequest) (*empty.Empty, error)
	InitRabbitForChain(context.Context, *ChainRequest) (*empty.Empty, error)
	StopReceivingRabbit(context.Context, *StopRequest) (*empty.Empty, error)
	Consume(*ConsumeRequest, SideCar_ConsumeServer) error
	ChainConsume(*ConsumeRequest, SideCar_ChainConsumeServer) error
	SendRequestApproval(context.Context, *RequestApproval) (*empty.Empty, error)
	SendValidationResponse(context.Context, *ValidationResponse) (*empty.Empty, error)
	SendCompositionRequest(context.Context, *CompositionRequest) (*empty.Empty, error)
	SendSqlDataRequest(context.Context, *SqlDataRequest) (*empty.Empty, error)
	SendPolicyUpdate(context.Context, *PolicyUpdate) (*empty.Empty, error)
	SendTest(context.Context, *SqlDataRequest) (*empty.Empty, error)
	SendMicroserviceComm(context.Context, *MicroserviceCommunication) (*empty.Empty, error)
	CreateQueue(context.Context, *QueueInfo) (*empty.Empty, error)
	DeleteQueue(context.Context, *QueueInfo) (*empty.Empty, error)
	SendRequestApprovalResponse(context.Context, *RequestApprovalResponse) (*empty.Empty, error)
	SendRequestApprovalRequest(context.Context, *RequestApproval) (*empty.Empty, error)
	mustEmbedUnimplementedSideCarServer()
}

// UnimplementedSideCarServer must be embedded to have forward compatible implementations.
type UnimplementedSideCarServer struct {
}

func (UnimplementedSideCarServer) InitRabbitMq(context.Context, *InitRequest) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method InitRabbitMq not implemented")
}
func (UnimplementedSideCarServer) InitRabbitForChain(context.Context, *ChainRequest) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method InitRabbitForChain not implemented")
}
func (UnimplementedSideCarServer) StopReceivingRabbit(context.Context, *StopRequest) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method StopReceivingRabbit not implemented")
}
func (UnimplementedSideCarServer) Consume(*ConsumeRequest, SideCar_ConsumeServer) error {
	return status.Errorf(codes.Unimplemented, "method Consume not implemented")
}
func (UnimplementedSideCarServer) ChainConsume(*ConsumeRequest, SideCar_ChainConsumeServer) error {
	return status.Errorf(codes.Unimplemented, "method ChainConsume not implemented")
}
func (UnimplementedSideCarServer) SendRequestApproval(context.Context, *RequestApproval) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendRequestApproval not implemented")
}
func (UnimplementedSideCarServer) SendValidationResponse(context.Context, *ValidationResponse) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendValidationResponse not implemented")
}
func (UnimplementedSideCarServer) SendCompositionRequest(context.Context, *CompositionRequest) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendCompositionRequest not implemented")
}
func (UnimplementedSideCarServer) SendSqlDataRequest(context.Context, *SqlDataRequest) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendSqlDataRequest not implemented")
}
func (UnimplementedSideCarServer) SendPolicyUpdate(context.Context, *PolicyUpdate) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendPolicyUpdate not implemented")
}
func (UnimplementedSideCarServer) SendTest(context.Context, *SqlDataRequest) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendTest not implemented")
}
func (UnimplementedSideCarServer) SendMicroserviceComm(context.Context, *MicroserviceCommunication) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendMicroserviceComm not implemented")
}
func (UnimplementedSideCarServer) CreateQueue(context.Context, *QueueInfo) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method CreateQueue not implemented")
}
func (UnimplementedSideCarServer) DeleteQueue(context.Context, *QueueInfo) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method DeleteQueue not implemented")
}
func (UnimplementedSideCarServer) SendRequestApprovalResponse(context.Context, *RequestApprovalResponse) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendRequestApprovalResponse not implemented")
}
func (UnimplementedSideCarServer) SendRequestApprovalRequest(context.Context, *RequestApproval) (*empty.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendRequestApprovalRequest not implemented")
}
func (UnimplementedSideCarServer) mustEmbedUnimplementedSideCarServer() {}

// UnsafeSideCarServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to SideCarServer will
// result in compilation errors.
type UnsafeSideCarServer interface {
	mustEmbedUnimplementedSideCarServer()
}

func RegisterSideCarServer(s grpc.ServiceRegistrar, srv SideCarServer) {
	s.RegisterService(&SideCar_ServiceDesc, srv)
}

func _SideCar_InitRabbitMq_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(InitRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).InitRabbitMq(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_InitRabbitMq_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).InitRabbitMq(ctx, req.(*InitRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_InitRabbitForChain_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ChainRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).InitRabbitForChain(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_InitRabbitForChain_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).InitRabbitForChain(ctx, req.(*ChainRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_StopReceivingRabbit_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(StopRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).StopReceivingRabbit(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_StopReceivingRabbit_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).StopReceivingRabbit(ctx, req.(*StopRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_Consume_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ConsumeRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(SideCarServer).Consume(m, &sideCarConsumeServer{ServerStream: stream})
}

type SideCar_ConsumeServer interface {
	Send(*SideCarMessage) error
	grpc.ServerStream
}

type sideCarConsumeServer struct {
	grpc.ServerStream
}

func (x *sideCarConsumeServer) Send(m *SideCarMessage) error {
	return x.ServerStream.SendMsg(m)
}

func _SideCar_ChainConsume_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ConsumeRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(SideCarServer).ChainConsume(m, &sideCarChainConsumeServer{ServerStream: stream})
}

type SideCar_ChainConsumeServer interface {
	Send(*SideCarMessage) error
	grpc.ServerStream
}

type sideCarChainConsumeServer struct {
	grpc.ServerStream
}

func (x *sideCarChainConsumeServer) Send(m *SideCarMessage) error {
	return x.ServerStream.SendMsg(m)
}

func _SideCar_SendRequestApproval_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RequestApproval)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendRequestApproval(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendRequestApproval_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendRequestApproval(ctx, req.(*RequestApproval))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendValidationResponse_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ValidationResponse)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendValidationResponse(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendValidationResponse_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendValidationResponse(ctx, req.(*ValidationResponse))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendCompositionRequest_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CompositionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendCompositionRequest(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendCompositionRequest_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendCompositionRequest(ctx, req.(*CompositionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendSqlDataRequest_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SqlDataRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendSqlDataRequest(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendSqlDataRequest_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendSqlDataRequest(ctx, req.(*SqlDataRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendPolicyUpdate_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PolicyUpdate)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendPolicyUpdate(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendPolicyUpdate_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendPolicyUpdate(ctx, req.(*PolicyUpdate))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendTest_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SqlDataRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendTest(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendTest_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendTest(ctx, req.(*SqlDataRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendMicroserviceComm_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MicroserviceCommunication)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendMicroserviceComm(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendMicroserviceComm_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendMicroserviceComm(ctx, req.(*MicroserviceCommunication))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_CreateQueue_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(QueueInfo)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).CreateQueue(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_CreateQueue_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).CreateQueue(ctx, req.(*QueueInfo))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_DeleteQueue_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(QueueInfo)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).DeleteQueue(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_DeleteQueue_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).DeleteQueue(ctx, req.(*QueueInfo))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendRequestApprovalResponse_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RequestApprovalResponse)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendRequestApprovalResponse(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendRequestApprovalResponse_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendRequestApprovalResponse(ctx, req.(*RequestApprovalResponse))
	}
	return interceptor(ctx, in, info, handler)
}

func _SideCar_SendRequestApprovalRequest_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RequestApproval)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(SideCarServer).SendRequestApprovalRequest(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: SideCar_SendRequestApprovalRequest_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(SideCarServer).SendRequestApprovalRequest(ctx, req.(*RequestApproval))
	}
	return interceptor(ctx, in, info, handler)
}

// SideCar_ServiceDesc is the grpc.ServiceDesc for SideCar service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var SideCar_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dynamos.SideCar",
	HandlerType: (*SideCarServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "InitRabbitMq",
			Handler:    _SideCar_InitRabbitMq_Handler,
		},
		{
			MethodName: "InitRabbitForChain",
			Handler:    _SideCar_InitRabbitForChain_Handler,
		},
		{
			MethodName: "StopReceivingRabbit",
			Handler:    _SideCar_StopReceivingRabbit_Handler,
		},
		{
			MethodName: "SendRequestApproval",
			Handler:    _SideCar_SendRequestApproval_Handler,
		},
		{
			MethodName: "SendValidationResponse",
			Handler:    _SideCar_SendValidationResponse_Handler,
		},
		{
			MethodName: "SendCompositionRequest",
			Handler:    _SideCar_SendCompositionRequest_Handler,
		},
		{
			MethodName: "SendSqlDataRequest",
			Handler:    _SideCar_SendSqlDataRequest_Handler,
		},
		{
			MethodName: "SendPolicyUpdate",
			Handler:    _SideCar_SendPolicyUpdate_Handler,
		},
		{
			MethodName: "SendTest",
			Handler:    _SideCar_SendTest_Handler,
		},
		{
			MethodName: "SendMicroserviceComm",
			Handler:    _SideCar_SendMicroserviceComm_Handler,
		},
		{
			MethodName: "CreateQueue",
			Handler:    _SideCar_CreateQueue_Handler,
		},
		{
			MethodName: "DeleteQueue",
			Handler:    _SideCar_DeleteQueue_Handler,
		},
		{
			MethodName: "SendRequestApprovalResponse",
			Handler:    _SideCar_SendRequestApprovalResponse_Handler,
		},
		{
			MethodName: "SendRequestApprovalRequest",
			Handler:    _SideCar_SendRequestApprovalRequest_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Consume",
			Handler:       _SideCar_Consume_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "ChainConsume",
			Handler:       _SideCar_ChainConsume_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "rabbitMQ.proto",
}
