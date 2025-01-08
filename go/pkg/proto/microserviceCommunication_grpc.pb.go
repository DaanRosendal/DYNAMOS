// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.5.1
// - protoc             v5.28.1
// source: microserviceCommunication.proto

package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	Microservice_SendData_FullMethodName = "/dynamos.Microservice/SendData"
)

// MicroserviceClient is the client API for Microservice service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// The sidecar definition.
type MicroserviceClient interface {
	SendData(ctx context.Context, in *MicroserviceCommunication, opts ...grpc.CallOption) (*ContinueReceiving, error)
}

type microserviceClient struct {
	cc grpc.ClientConnInterface
}

func NewMicroserviceClient(cc grpc.ClientConnInterface) MicroserviceClient {
	return &microserviceClient{cc}
}

func (c *microserviceClient) SendData(ctx context.Context, in *MicroserviceCommunication, opts ...grpc.CallOption) (*ContinueReceiving, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ContinueReceiving)
	err := c.cc.Invoke(ctx, Microservice_SendData_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// MicroserviceServer is the server API for Microservice service.
// All implementations must embed UnimplementedMicroserviceServer
// for forward compatibility.
//
// The sidecar definition.
type MicroserviceServer interface {
	SendData(context.Context, *MicroserviceCommunication) (*ContinueReceiving, error)
	mustEmbedUnimplementedMicroserviceServer()
}

// UnimplementedMicroserviceServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedMicroserviceServer struct{}

func (UnimplementedMicroserviceServer) SendData(context.Context, *MicroserviceCommunication) (*ContinueReceiving, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendData not implemented")
}
func (UnimplementedMicroserviceServer) mustEmbedUnimplementedMicroserviceServer() {}
func (UnimplementedMicroserviceServer) testEmbeddedByValue()                      {}

// UnsafeMicroserviceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to MicroserviceServer will
// result in compilation errors.
type UnsafeMicroserviceServer interface {
	mustEmbedUnimplementedMicroserviceServer()
}

func RegisterMicroserviceServer(s grpc.ServiceRegistrar, srv MicroserviceServer) {
	// If the following call pancis, it indicates UnimplementedMicroserviceServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&Microservice_ServiceDesc, srv)
}

func _Microservice_SendData_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MicroserviceCommunication)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MicroserviceServer).SendData(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Microservice_SendData_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MicroserviceServer).SendData(ctx, req.(*MicroserviceCommunication))
	}
	return interceptor(ctx, in, info, handler)
}

// Microservice_ServiceDesc is the grpc.ServiceDesc for Microservice service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Microservice_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dynamos.Microservice",
	HandlerType: (*MicroserviceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "SendData",
			Handler:    _Microservice_SendData_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "microserviceCommunication.proto",
}
