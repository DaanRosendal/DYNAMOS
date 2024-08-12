# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rabbitMQ.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
import generic_pb2 as generic__pb2
import microserviceCommunication_pb2 as microserviceCommunication__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0erabbitMQ.proto\x12\x07\x64ynamos\x1a\x1bgoogle/protobuf/empty.proto\x1a\x19google/protobuf/any.proto\x1a\rgeneric.proto\x1a\x1fmicroserviceCommunication.proto\"S\n\x0bInitRequest\x12\x14\n\x0cservice_name\x18\x01 \x01(\t\x12\x13\n\x0brouting_key\x18\x02 \x01(\t\x12\x19\n\x11queue_auto_delete\x18\x03 \x01(\x08\"b\n\x0c\x43hainRequest\x12\x14\n\x0cservice_name\x18\x01 \x01(\t\x12\x13\n\x0brouting_key\x18\x02 \x01(\t\x12\x19\n\x11queue_auto_delete\x18\x03 \x01(\x08\x12\x0c\n\x04port\x18\x04 \x01(\r\"\r\n\x0bStopRequest\"Y\n\tQueueInfo\x12\x12\n\nqueue_name\x18\x01 \x01(\t\x12\x13\n\x0b\x61uto_delete\x18\x02 \x01(\x08\x12\x11\n\tuser_name\x18\x03 \x01(\t\x12\x10\n\x08job_name\x18\x04 \x01(\t\"6\n\x0e\x43onsumeRequest\x12\x12\n\nqueue_name\x18\x01 \x01(\t\x12\x10\n\x08\x61uto_ack\x18\x02 \x01(\x08\"\xa6\x01\n\x0eSideCarMessage\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\"\n\x04\x62ody\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any\x12\x33\n\x06traces\x18\x03 \x03(\x0b\x32#.dynamos.SideCarMessage.TracesEntry\x1a-\n\x0bTracesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"3\n\x04\x41uth\x12\x14\n\x0c\x61\x63\x63\x65ss_token\x18\x01 \x01(\t\x12\x15\n\rrefresh_token\x18\x02 \x01(\t\"=\n\x0c\x44\x61taProvider\x12\x12\n\narchetypes\x18\x01 \x03(\t\x12\x19\n\x11\x63ompute_providers\x18\x02 \x03(\t\"+\n\x15UserAllowedArchetypes\x12\x12\n\narchetypes\x18\x01 \x03(\t\"\xb3\x01\n\x0eUserArchetypes\x12\x11\n\tuser_name\x18\x01 \x01(\t\x12;\n\narchetypes\x18\x02 \x03(\x0b\x32\'.dynamos.UserArchetypes.ArchetypesEntry\x1aQ\n\x0f\x41rchetypesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x1e.dynamos.UserAllowedArchetypes:\x02\x38\x01\"\xed\x03\n\x12ValidationResponse\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x14\n\x0crequest_type\x18\x02 \x01(\t\x12P\n\x13valid_dataproviders\x18\x03 \x03(\x0b\x32\x33.dynamos.ValidationResponse.ValidDataprovidersEntry\x12\x1d\n\x15invalid_dataproviders\x18\x04 \x03(\t\x12\x1b\n\x04\x61uth\x18\x05 \x01(\x0b\x32\r.dynamos.Auth\x12\x1b\n\x04user\x18\x06 \x01(\x0b\x32\r.dynamos.User\x12\x18\n\x10request_approved\x18\x07 \x01(\x08\x12\x31\n\x10valid_archetypes\x18\x08 \x01(\x0b\x32\x17.dynamos.UserArchetypes\x12\x39\n\x07options\x18\t \x03(\x0b\x32(.dynamos.ValidationResponse.OptionsEntry\x1aP\n\x17ValidDataprovidersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.dynamos.DataProvider:\x02\x38\x01\x1a.\n\x0cOptionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\"%\n\x04User\x12\n\n\x02id\x18\x01 \x01(\t\x12\x11\n\tuser_name\x18\x02 \x01(\t\"\xd7\x01\n\x0fRequestApproval\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x1b\n\x04user\x18\x02 \x01(\x0b\x32\r.dynamos.User\x12\x16\n\x0e\x64\x61ta_providers\x18\x03 \x03(\t\x12\x19\n\x11\x64\x65stination_queue\x18\x04 \x01(\t\x12\x36\n\x07options\x18\x05 \x03(\x0b\x32%.dynamos.RequestApproval.OptionsEntry\x1a.\n\x0cOptionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\"\xc9\x02\n\x17RequestApprovalResponse\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x1b\n\x04user\x18\x02 \x01(\x0b\x32\r.dynamos.User\x12\x1b\n\x04\x61uth\x18\x03 \x01(\x0b\x32\r.dynamos.Auth\x12W\n\x14\x61uthorized_providers\x18\x04 \x03(\x0b\x32\x39.dynamos.RequestApprovalResponse.AuthorizedProvidersEntry\x12\x0e\n\x06job_id\x18\x05 \x01(\t\x12\r\n\x05\x65rror\x18\x06 \x01(\t\x12\x32\n\x10request_metadata\x18\x07 \x01(\x0b\x32\x18.dynamos.RequestMetadata\x1a:\n\x18\x41uthorizedProvidersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xbf\x01\n\x0cPolicyUpdate\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x1b\n\x04user\x18\x02 \x01(\x0b\x32\r.dynamos.User\x12\x16\n\x0e\x64\x61ta_providers\x18\x03 \x03(\t\x12\x32\n\x10request_metadata\x18\x04 \x01(\x0b\x32\x18.dynamos.RequestMetadata\x12\x38\n\x13validation_response\x18\x05 \x01(\x0b\x32\x1b.dynamos.ValidationResponse\"\xc8\x01\n\x12\x43ompositionRequest\x12\x14\n\x0c\x61rchetype_id\x18\x01 \x01(\t\x12\x14\n\x0crequest_type\x18\x02 \x01(\t\x12\x0c\n\x04role\x18\x03 \x01(\t\x12\x1b\n\x04user\x18\x04 \x01(\x0b\x32\r.dynamos.User\x12\x16\n\x0e\x64\x61ta_providers\x18\x05 \x03(\t\x12\x19\n\x11\x64\x65stination_queue\x18\x06 \x01(\t\x12\x10\n\x08job_name\x18\x07 \x01(\t\x12\x16\n\x0elocal_job_name\x18\x08 \x01(\t\"\xfb\x02\n\x0eSqlDataRequest\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\r\n\x05query\x18\x02 \x01(\t\x12\x11\n\talgorithm\x18\x03 \x01(\t\x12H\n\x11\x61lgorithm_columns\x18\x04 \x03(\x0b\x32-.dynamos.SqlDataRequest.AlgorithmColumnsEntry\x12\x1b\n\x04user\x18\x05 \x01(\x0b\x32\r.dynamos.User\x12\x32\n\x10request_metadata\x18\x06 \x01(\x0b\x32\x18.dynamos.RequestMetadata\x12\x35\n\x07options\x18\x07 \x03(\x0b\x32$.dynamos.SqlDataRequest.OptionsEntry\x1a\x37\n\x15\x41lgorithmColumnsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a.\n\x0cOptionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\x32\x95\t\n\x07SideCar\x12>\n\x0cInitRabbitMq\x12\x14.dynamos.InitRequest\x1a\x16.google.protobuf.Empty\"\x00\x12\x45\n\x12InitRabbitForChain\x12\x15.dynamos.ChainRequest\x1a\x16.google.protobuf.Empty\"\x00\x12\x45\n\x13StopReceivingRabbit\x12\x14.dynamos.StopRequest\x1a\x16.google.protobuf.Empty\"\x00\x12?\n\x07\x43onsume\x12\x17.dynamos.ConsumeRequest\x1a\x17.dynamos.SideCarMessage\"\x00\x30\x01\x12\x44\n\x0c\x43hainConsume\x12\x17.dynamos.ConsumeRequest\x1a\x17.dynamos.SideCarMessage\"\x00\x30\x01\x12I\n\x13SendRequestApproval\x12\x18.dynamos.RequestApproval\x1a\x16.google.protobuf.Empty\"\x00\x12O\n\x16SendValidationResponse\x12\x1b.dynamos.ValidationResponse\x1a\x16.google.protobuf.Empty\"\x00\x12O\n\x16SendCompositionRequest\x12\x1b.dynamos.CompositionRequest\x1a\x16.google.protobuf.Empty\"\x00\x12G\n\x12SendSqlDataRequest\x12\x17.dynamos.SqlDataRequest\x1a\x16.google.protobuf.Empty\"\x00\x12\x43\n\x10SendPolicyUpdate\x12\x15.dynamos.PolicyUpdate\x1a\x16.google.protobuf.Empty\"\x00\x12=\n\x08SendTest\x12\x17.dynamos.SqlDataRequest\x1a\x16.google.protobuf.Empty\"\x00\x12T\n\x14SendMicroserviceComm\x12\".dynamos.MicroserviceCommunication\x1a\x16.google.protobuf.Empty\"\x00\x12;\n\x0b\x43reateQueue\x12\x12.dynamos.QueueInfo\x1a\x16.google.protobuf.Empty\"\x00\x12;\n\x0b\x44\x65leteQueue\x12\x12.dynamos.QueueInfo\x1a\x16.google.protobuf.Empty\"\x00\x12Y\n\x1bSendRequestApprovalResponse\x12 .dynamos.RequestApprovalResponse\x1a\x16.google.protobuf.Empty\"\x00\x12P\n\x1aSendRequestApprovalRequest\x12\x18.dynamos.RequestApproval\x1a\x16.google.protobuf.Empty\"\x00\x42\'Z%github.com/Jorrit05/DYNAMOS/pkg/protob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rabbitMQ_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z%github.com/Jorrit05/DYNAMOS/pkg/proto'
  _SIDECARMESSAGE_TRACESENTRY._options = None
  _SIDECARMESSAGE_TRACESENTRY._serialized_options = b'8\001'
  _USERARCHETYPES_ARCHETYPESENTRY._options = None
  _USERARCHETYPES_ARCHETYPESENTRY._serialized_options = b'8\001'
  _VALIDATIONRESPONSE_VALIDDATAPROVIDERSENTRY._options = None
  _VALIDATIONRESPONSE_VALIDDATAPROVIDERSENTRY._serialized_options = b'8\001'
  _VALIDATIONRESPONSE_OPTIONSENTRY._options = None
  _VALIDATIONRESPONSE_OPTIONSENTRY._serialized_options = b'8\001'
  _REQUESTAPPROVAL_OPTIONSENTRY._options = None
  _REQUESTAPPROVAL_OPTIONSENTRY._serialized_options = b'8\001'
  _REQUESTAPPROVALRESPONSE_AUTHORIZEDPROVIDERSENTRY._options = None
  _REQUESTAPPROVALRESPONSE_AUTHORIZEDPROVIDERSENTRY._serialized_options = b'8\001'
  _SQLDATAREQUEST_ALGORITHMCOLUMNSENTRY._options = None
  _SQLDATAREQUEST_ALGORITHMCOLUMNSENTRY._serialized_options = b'8\001'
  _SQLDATAREQUEST_OPTIONSENTRY._options = None
  _SQLDATAREQUEST_OPTIONSENTRY._serialized_options = b'8\001'
  _globals['_INITREQUEST']._serialized_start=131
  _globals['_INITREQUEST']._serialized_end=214
  _globals['_CHAINREQUEST']._serialized_start=216
  _globals['_CHAINREQUEST']._serialized_end=314
  _globals['_STOPREQUEST']._serialized_start=316
  _globals['_STOPREQUEST']._serialized_end=329
  _globals['_QUEUEINFO']._serialized_start=331
  _globals['_QUEUEINFO']._serialized_end=420
  _globals['_CONSUMEREQUEST']._serialized_start=422
  _globals['_CONSUMEREQUEST']._serialized_end=476
  _globals['_SIDECARMESSAGE']._serialized_start=479
  _globals['_SIDECARMESSAGE']._serialized_end=645
  _globals['_SIDECARMESSAGE_TRACESENTRY']._serialized_start=600
  _globals['_SIDECARMESSAGE_TRACESENTRY']._serialized_end=645
  _globals['_AUTH']._serialized_start=647
  _globals['_AUTH']._serialized_end=698
  _globals['_DATAPROVIDER']._serialized_start=700
  _globals['_DATAPROVIDER']._serialized_end=761
  _globals['_USERALLOWEDARCHETYPES']._serialized_start=763
  _globals['_USERALLOWEDARCHETYPES']._serialized_end=806
  _globals['_USERARCHETYPES']._serialized_start=809
  _globals['_USERARCHETYPES']._serialized_end=988
  _globals['_USERARCHETYPES_ARCHETYPESENTRY']._serialized_start=907
  _globals['_USERARCHETYPES_ARCHETYPESENTRY']._serialized_end=988
  _globals['_VALIDATIONRESPONSE']._serialized_start=991
  _globals['_VALIDATIONRESPONSE']._serialized_end=1484
  _globals['_VALIDATIONRESPONSE_VALIDDATAPROVIDERSENTRY']._serialized_start=1356
  _globals['_VALIDATIONRESPONSE_VALIDDATAPROVIDERSENTRY']._serialized_end=1436
  _globals['_VALIDATIONRESPONSE_OPTIONSENTRY']._serialized_start=1438
  _globals['_VALIDATIONRESPONSE_OPTIONSENTRY']._serialized_end=1484
  _globals['_USER']._serialized_start=1486
  _globals['_USER']._serialized_end=1523
  _globals['_REQUESTAPPROVAL']._serialized_start=1526
  _globals['_REQUESTAPPROVAL']._serialized_end=1741
  _globals['_REQUESTAPPROVAL_OPTIONSENTRY']._serialized_start=1438
  _globals['_REQUESTAPPROVAL_OPTIONSENTRY']._serialized_end=1484
  _globals['_REQUESTAPPROVALRESPONSE']._serialized_start=1744
  _globals['_REQUESTAPPROVALRESPONSE']._serialized_end=2073
  _globals['_REQUESTAPPROVALRESPONSE_AUTHORIZEDPROVIDERSENTRY']._serialized_start=2015
  _globals['_REQUESTAPPROVALRESPONSE_AUTHORIZEDPROVIDERSENTRY']._serialized_end=2073
  _globals['_POLICYUPDATE']._serialized_start=2076
  _globals['_POLICYUPDATE']._serialized_end=2267
  _globals['_COMPOSITIONREQUEST']._serialized_start=2270
  _globals['_COMPOSITIONREQUEST']._serialized_end=2470
  _globals['_SQLDATAREQUEST']._serialized_start=2473
  _globals['_SQLDATAREQUEST']._serialized_end=2852
  _globals['_SQLDATAREQUEST_ALGORITHMCOLUMNSENTRY']._serialized_start=2749
  _globals['_SQLDATAREQUEST_ALGORITHMCOLUMNSENTRY']._serialized_end=2804
  _globals['_SQLDATAREQUEST_OPTIONSENTRY']._serialized_start=1438
  _globals['_SQLDATAREQUEST_OPTIONSENTRY']._serialized_end=1484
  _globals['_SIDECAR']._serialized_start=2855
  _globals['_SIDECAR']._serialized_end=4028
# @@protoc_insertion_point(module_scope)
