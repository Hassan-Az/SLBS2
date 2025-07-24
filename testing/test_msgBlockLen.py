

message = 'Hello this is a text message and a big sentence too'
msg1 = [ord(c) for c in message]
msg_length = len(msg1)

# here the numberOfBlock is static, we will be needing dynamic to improve the system
numberOfBlocks = 4
msgBlock = int(msg_length/numberOfBlocks)
lastMsgBlock = msg_length - msgBlock * numberOfBlocks

print(msg_length, msgBlock, lastMsgBlock)