# Copyright 2023 Axini B.V. https://www.axini.com, see: LICENSE.txt.
# frozen_string_literal: true

# WebSocket::Driver requires the TCPSocket object to have an attribute url.
module OpenSSL
  module SSL
    class SSLSocket
      attr_accessor :url
    end
  end
end

# The DynamosConnection holds the WebSocket connection with the standalone
# DYNAMOS SUT
class Dynamos::Connection
  def initialize(url, handler)
    @url     = url
    @handler = handler
    @socket  = nil
    @driver  = nil
    @queue_handler = nil
  end

  # Connect to AMP's plugin adapter broker and register WebSocket callbacks.
  def connect
    @queue_handler = RabbitMQHandler.new(@handler, 'mbt_parallel_queue')
  end

  # Close the given websocket with the given response close code and reason.
  # @param [Integer] code
  # @param [String] reason
  def close(reason: nil, code: 1000)
    @queue_handler&.close
  end

  def send(message)
    @queue_handler.send_message(message)
  end
end
