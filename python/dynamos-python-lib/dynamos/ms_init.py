import os
import logging
from .logger import InitLogger
from .grpc_client import GRPCClient
from .grpc_server import GRPCServer
import microserviceCommunication_pb2 as msCommTypes
from typing import Callable, Dict
from google.protobuf.empty_pb2 import Empty


logger = InitLogger()

class Configuration:
    def __init__(self,
                 port,
                 first_service,
                 last_service,
                 service_name,
                 ms_message_handler : Callable[[msCommTypes.MicroserviceCommunication], Empty]):
        self.Port = port
        self.first_service = first_service
        self.last_service = last_service
        self.service_name = service_name
        self.rabbit_msg_client = None
        self.grpc_server = None
        self.next_client = None
        self.ms_message_handler = ms_message_handler

    def init_sidecar_messaging(self, receive_mutex):
        pass  # Implement sidecar messaging initialization


def NewConfiguration(service_name,
                      grpc_addr,
                      ms_message_handler:  Callable[[msCommTypes.MicroserviceCommunication],Empty ]):
    try:
        port = int(os.getenv("DESIGNATED_GRPC_PORT"))
    except ValueError:
        raise ValueError("Error determining port number")

    try:
        # If first service, setup incoming RabbitMQ channel
        first_service = int(os.getenv("FIRST"))
    except ValueError:
        raise ValueError("Error determining first service")

    try:
        # If last service, send outgoing message to Sidecar for rabbitMQ to process
        last_service = int(os.getenv("LAST"))
    except ValueError:
        raise ValueError("Error determining last service")

    logger.debug(f"NewConfiguration {service_name}, port: {port}, last_service: {last_service}")

    conf = Configuration(
        port=port,
        first_service=first_service > 0,
        last_service=last_service > 0,
        service_name=service_name,
        ms_message_handler=ms_message_handler
    )

    if conf.first_service:
        # First and last, connect to sidecar for processing and final destination
        conf.grpc_server = GRPCServer(grpc_addr + str(conf.Port), ms_message_handler)
        conf.rabbit_msg_client = GRPCClient(grpc_addr + os.getenv("SIDECAR_PORT"), service_name)
        if conf.last_service:
            conf.next_client = conf.rabbit_msg_client
        else:
            conf.next_client = GRPCClient(grpc_addr + str(conf.Port + 1), service_name)

        conf.rabbit_msg_client.rabbit.initialize_rabbit(service_name, conf.Port)

    elif conf.last_service:
        # Last service, connect to sidecar as final destination and start own server to receive from previous MS
        conf.grpc_server = GRPCServer(grpc_addr + str(conf.Port), ms_message_handler)
        conf.next_client = GRPCClient(grpc_addr + os.getenv("SIDECAR_PORT"), service_name)
    else:
        conf.grpc_server = GRPCServer(grpc_addr + str(conf.Port), ms_message_handler)
        conf.next_client = GRPCClient(grpc_addr + str(conf.Port + 1), service_name)

    return conf
