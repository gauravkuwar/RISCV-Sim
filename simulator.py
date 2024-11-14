import os
import argparse
import copy

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space reason, we keep it as this large number, but the memory is still 32-bit addressable.

def flipBits(val): 
    return ''.join(map(lambda x: str(int(not int(x))), val))

def b2i(x): 
    return int(x, 2) # binary to int

def i2b(x, bit_num=32): 
    return bin(x)[2:].zfill(bit_num) # int to binary 32-bit

def b2twos(val): 
    # converts 2s complement 32-bit binary to decimal
    if not val: return '0'*32
    if val[0] == '0': 
        return b2i(val)
    else: 
        return -(b2i(flipBits(val[1:])) + 1)

def twos2b(val): 
    # converts decimal to 2s complement 32-bit binary
    if val >= 0: 
        return i2b(val)
    else: # handle negatives
        return '1' + flipBits(i2b((-val) - 1, bit_num=31))

class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        
        with open(os.path.join(ioDir, "imem.txt")) as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        # read instruction memory
        # return 32 bit hex val
        return ''.join(self.IMem[ReadAddress:ReadAddress+4])
          
class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(os.path.join(ioDir, "dmem.txt")) as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        self.DMem += ['0' * 8] * (MemSize - len(self.DMem)) # make it MemSize

    def readInstr(self, ReadAddress):
        # read data memory
        # return 32 bit hex val
        return b2twos(''.join(self.DMem[ReadAddress:ReadAddress+4]))
        
    def writeDataMem(self, Address, WriteData):
        # write data into byte addressable memory
        dataBits = twos2b(WriteData)
        for i in range(4):
            self.DMem[Address+i] = dataBits[8*i:8*(i+1)]
                     
    def outputDataMem(self):
        resPath = os.path.join(self.ioDir, f"{self.id}_DMEMResult.txt")
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [i2b(0) for i in range(32)]
    
    def readRF(self, Reg_addr):
        # Fill in
        return b2twos(self.Registers[Reg_addr])
    
    def writeRF(self, Reg_addr, Wrt_reg_data):
        # Fill in
        if Reg_addr != 0: # Zero register does not change
            self.Registers[Reg_addr] = twos2b(Wrt_reg_data)
         
    def outputRF(self, cycle):
        op = ["-"*70+"\n", "State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([str(val)+"\n" for val in self.Registers])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        """
        For Single Stage these hashmaps are not really needed, but I use it for consistency
        For Five Stage Pipeline, acts for the registers needed for forwarding
        IF = Just to store PC
        ID = IF/ID 
        EX = ID/EX
        MEM = EX/MEM
        WB = MEM/WB
        """

        self.IF = {"nop": False, "PC": 0}
        self.ID = {"nop": False, "PC": 0, "Instr": 0}
        self.EX = {"nop": False, "PC": 0, "rs1": 0, "rs2": 0, "aluControlBits": 0, "Imm": 0,"ALUSrc": 0, "ALUOp": 0, 
                   "Branch": 0, "ALUResult": 0, "Zero": 0, "MemRead": 0, "MemWrite": 0, "Wrt_reg_addr": 0,
                   "MemToReg": 0, "RegWrite": 0, "Reg_rs1": 0, "Reg_rs2": 0}
        self.MEM = {"nop": False, "PC": 0, "ALUResult": 0, "MemRead": 0, "MemWrite": 0, "Wrt_reg_addr": 0, "MemToReg": 0, "RegWrite": 0, "rs2": 0, "Branch": 0}
        self.WB = {"nop": False, "ALUResult": 0, "ReadData": 0, "MemToReg": 0, "Wrt_reg_addr": 0, "RegWrite": 0, "dataToWrite": 0}

class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.instrCount = 1
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem
    
    def decodeInstr(self):
        instr = self.state.ID["Instr"]
        decodedInstr = {
            'rs1': instr[12:17],
            'rs2': instr[7:12],
            'opcode': instr[25:32],
            'rd': instr[20:25],
            'funct7': instr[0:7],
            'funct3': instr[17:20]
        }
        return decodedInstr

    def controlUnit(self, opcode):
        out = {x:0 for x in ('RegWrite', 'ALUOp', 'ALUSrc', 'MemToReg', 'MemRead', 'MemWrite', 'Branch')}
        if opcode == 0x33: # R
            out['RegWrite'] = 1
            out['ALUOp'] = 0b10

        if opcode == 0x13: # I
            out['RegWrite'] = 1
            out['ALUSrc'] = 1

        if opcode == 0x3: # lw
            out['ALUSrc'] = 1
            out['MemToReg'] = 1
            out['RegWrite'] = 1
            out['MemRead'] = 1

        if opcode == 0x23: # sw
            out['ALUSrc'] = 1
            out['MemWrite'] = 1

        if opcode == 0x63: # beq/bne
            out['Branch'] = 1
            out['ALUOp'] = 0b01

        if opcode == 0x6f: # jal
            out['RegWrite'] = 1
            out['Branch'] = 1

        return out


    def immGenUnit(self):
        instr = self.state.ID["Instr"]
        opcode = b2i(instr[25:32])

        # RISC-V Instruction Set Manual Page 17
        # immU = instr[0:20] + ('0' * 12)
        res = i2b(0)
        if opcode in (0x03, 0x13): # imm-I
            res = (instr[0] * 21) + instr[1:12]
        if opcode == 0x23: # imm-S
            res = (instr[0] * 21) + instr[1:7] + instr[20:25]
        if opcode == 0x63: # imm-B
            res = (instr[0] * 20) + instr[24] + instr[1:7] + instr[20:24] + '0'
        if opcode == 0x6F: # imm-J
            res = (instr[0] * 12) + instr[12:20] + instr[11] + instr[1:7] + instr[7:11] + '0'
        
        return b2twos(res)
    
    def ALUControlUnit(self, ALUOp, funct3, funct7_1bit):
        if ALUOp == 0b10 and funct3 == 0b000: # R and add/sub
            # if funct3 == 0b000: # add and sub
            return 0b0010 if funct7_1bit == 0 else 0b0110

        if ALUOp == 0b01: 
            # beq if funct3 == 0x1 else bne
            # I used sub to check zero for beq
            # and use xor to check zero for bne
            return 0b0110 if funct3 == 0b000 else 0b0111
        
        if funct3 == 0b111: return 0b0000 # and
        if funct3 == 0b110: return 0b0001 # or
        if funct3 == 0b100: return 0b0111 # xor
        return 0b0010 # rest is add
        
    def ALU(self, in1, in2, controlBits):
        res, zero = 0, 1
        # by setting zero to 1 by default
        # every non-branch operation don't care
        # but jal doesn't need to do anything for zero = 1
        if controlBits == 0b0010: # add
            res = in1 + in2
        if controlBits == 0b0110: # sub
            res = in1 - in2
            zero = int(res == 0) # check for beq
        if controlBits == 0b0000: # and
            res = in1 & in2
        if controlBits == 0b0001: # or
            res = in1 | in2
        if controlBits == 0b0111: # xor
            res = in1 ^ in2
            zero = int(res != 0) # check for bne

        return res, zero

class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(os.path.join(ioDir, "SS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_SS.txt")

    def step(self):
        # Your implementation
        # INSTRUCTION FETCH
        print("Cycle:", self.cycle, "PC:", self.state.IF["PC"])
        self.state.ID["Instr"] = self.ext_imem.readInstr(self.state.IF["PC"])

        # INSTRUCTION DECODE
        decodedInstr = self.decodeInstr()
        self.state.EX['rs1'] = self.myRF.readRF(b2i(decodedInstr['rs1']))
        self.state.EX['rs2'] = self.myRF.readRF(b2i(decodedInstr['rs2']))
        self.state.WB['Wrt_reg_addr'] = b2i(decodedInstr['rd'])

        CUOut = self.controlUnit(b2i(decodedInstr['opcode']))
        self.state.EX['ALUOp']      = CUOut['ALUOp']
        self.state.EX['ALUSrc']     = CUOut['ALUSrc']
        self.state.EX['Branch']     = CUOut['Branch']
        self.state.MEM['MemRead']   = CUOut['MemRead']
        self.state.MEM['MemWrite']  = CUOut['MemWrite']
        self.state.WB['MemToReg']   = CUOut['MemToReg']
        self.state.WB['RegWrite']   = CUOut['RegWrite']

        self.state.EX['Imm'] = self.immGenUnit()
        self.state.EX["aluControlBits"] = self.ALUControlUnit(self.state.EX['ALUOp'], 
                                                                b2i(decodedInstr['funct3']), 
                                                                b2i(decodedInstr['funct7'][1]))
        
        # EXECUTION
        in2 = self.state.EX['Imm'] if self.state.EX['ALUSrc'] else self.state.EX['rs2']
        self.state.EX['ALUResult'], self.state.EX['Zero'] = self.ALU(self.state.EX['rs1'], in2, self.state.EX["aluControlBits"])

        if b2twos(self.state.ID["Instr"]) != -1:
            self.instrCount += 1
            if self.state.EX['Zero'] and self.state.EX['Branch']:
                # no need to shift left 1 because I added extra 0 when decoding imm for J
                self.nextState.IF["PC"] += self.state.EX['Imm']
            else:
                self.nextState.IF["PC"] += 4
        else: # HALTED
            self.nextState.IF["nop"] = True

        # MEMORY
        # Memory Unit doesn't need a separate function - code is simple
        address = self.state.EX['ALUResult']
        if self.state.MEM['MemRead']: # lw
            self.state.WB['ReadData'] = self.ext_dmem.readInstr(address)
        elif self.state.EX['Branch']: # for jal instr
            self.state.EX['ALUResult'] = self.state.IF["PC"] + 4
        
        if self.state.MEM['MemWrite']: # sw
            self.ext_dmem.writeDataMem(address, self.state.EX['rs2'])
        
        # WRITE BACK
        if self.state.WB['RegWrite']:
            # if lw we write mem data to reg
            dataToWrite = self.state.WB['ReadData'] if self.state.WB['MemToReg'] else self.state.EX['ALUResult']
            self.myRF.writeRF(self.state.WB['Wrt_reg_addr'], dataToWrite)

        print("IF", self.state.IF)
        print("ID", self.state.IF)
        print("EX", self.state.EX)
        print("MEM", self.state.MEM)
        print("WB", self.state.WB)
        print()

        # self.halted = True
        if self.state.IF["nop"]:
            self.halted = True

        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
            
        self.state = copy.deepcopy(self.nextState) #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")
        
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(os.path.join(ioDir, "FS_"), imem, dmem)
        self.opFilePath = os.path.join(ioDir, "StateResult_FS.txt")
        self.state.IF['nop'] = False
        self.state.ID['nop'] = True
        self.state.EX['nop'] = True
        self.state.MEM['nop'] = True
        self.state.WB['nop'] = True
        self.nextState = copy.deepcopy(self.state)

    def forwardingUnit(self, EX_MEM_RegWrite, EX_MEM_RegisterRd,
                             MEM_WB_RegWrite, MEM_WB_RegisterRd,
                             ID_EX_RegistersRs1, ID_EX_RegistersRs2):
        
        forwardA, forwardB = 0b00, 0b00
        # EX Hazard
        if EX_MEM_RegWrite and EX_MEM_RegisterRd != 0 and ID_EX_RegistersRs1 == EX_MEM_RegisterRd:
            forwardA = 0b10
        if EX_MEM_RegWrite and EX_MEM_RegisterRd != 0 and ID_EX_RegistersRs2 == EX_MEM_RegisterRd:
            forwardB = 0b10

        # MEM Hazard
        if MEM_WB_RegWrite and MEM_WB_RegisterRd != 0 and ID_EX_RegistersRs1 == MEM_WB_RegisterRd:
            forwardA = 0b01
        if MEM_WB_RegWrite and MEM_WB_RegisterRd != 0 and ID_EX_RegistersRs2 == MEM_WB_RegisterRd:
            forwardB = 0b01

        return forwardA, forwardB
        
    def step(self):
        # Your implementation
        print("Cycle:", self.cycle)
                
        # --------------------- WB stage ---------------------
        if not self.state.WB["nop"]:
            self.instrCount += 1
            print("WB Stage <-----")
            print("WB:", self.state.WB)

            if self.state.WB['RegWrite']:
                # if lw we write mem data to reg
                self.state.WB['dataToWrite'] = self.state.WB['ReadData'] if self.state.WB['MemToReg'] else self.state.WB['ALUResult']
                print(f"reg addr: {self.state.WB['Wrt_reg_addr']}, writing data: {self.state.WB['dataToWrite']}")
                self.myRF.writeRF(self.state.WB['Wrt_reg_addr'], self.state.WB['dataToWrite'])

            print('-'*10)
        else:
            self.clearStage(self.state.WB)
        
        # --------------------- MEM stage --------------------
        if not self.state.MEM["nop"]:
            print("MEM Stage <-----")
            print("MEM:", self.state.MEM)

            # Memory Unit doesn't need a separate function - code is simple
            if self.state.MEM['MemRead']: # lw
                self.nextState.WB['ReadData'] = self.ext_dmem.readInstr(self.state.MEM['ALUResult'])
            elif self.state.MEM['Branch']: # for jal instr
                self.state.MEM['ALUResult'] = self.state.MEM["PC"] + 4
            if self.state.MEM['MemWrite']: # sw
                self.ext_dmem.writeDataMem(self.state.MEM['ALUResult'], self.state.MEM['rs2'])

            # pass forward
            self.nextState.WB['Wrt_reg_addr']   = self.state.MEM['Wrt_reg_addr']
            self.nextState.WB['MemToReg']       = self.state.MEM['MemToReg']
            self.nextState.WB['RegWrite']       = self.state.MEM['RegWrite']
            self.nextState.WB['Wrt_reg_addr']   = self.state.MEM['Wrt_reg_addr']
            self.nextState.WB['ALUResult']      = self.state.MEM['ALUResult']
            # The commented code can allow us to reduce the number of cycles, by skipping the WB cycle
            # for cases where we don't write back to register. Only by a few cycles tho, in cases
            # for example the sw is at the end. 
            self.nextState.WB["nop"] = False #if self.nextState.WB['RegWrite'] == 1 else True

            print('-'*10)

        else:
            self.clearStage(self.state.MEM)
            self.nextState.WB["nop"] = True

        # --------------------- EX stage ---------------------
        if not self.state.EX["nop"]:
            print("EX Stage <-----")
            print("EX:", self.state.EX)

            forwardA, forwardB = self.forwardingUnit(
                    EX_MEM_RegWrite=self.state.MEM['RegWrite'],
                    EX_MEM_RegisterRd=self.state.MEM['Wrt_reg_addr'],
                    MEM_WB_RegWrite=self.state.WB['RegWrite'],
                    MEM_WB_RegisterRd=self.state.WB['Wrt_reg_addr'],
                    ID_EX_RegistersRs1=self.state.EX['Reg_rs1'],
                    ID_EX_RegistersRs2=self.state.EX['Reg_rs2']
                    )
            
            if forwardA == 0b01:
                # From MEM read or prior ALU Result
                self.state.EX['rs1'] = self.state.WB['dataToWrite']
            elif forwardA == 0b10:
                self.state.EX['rs1'] = self.state.MEM['ALUResult'] # from prior ALU Result

            if forwardB == 0b01:
                # From MEM read or prior ALU Result
                self.state.EX['rs2'] = self.state.WB['dataToWrite']
            elif forwardB == 0b10:
                self.state.EX['rs2'] = self.state.MEM['ALUResult'] # from prior ALU Result
            # else stay the same

            # print(f"-- self.state.EX['rs2']: {self.state.EX['rs2']}, self.state.WB['dataToWrite']: {self.state.WB['dataToWrite']}, self.state.MEM['ALUResult']: {self.state.MEM['ALUResult']}")
            in2 = self.state.EX['Imm'] if self.state.EX['ALUSrc'] else self.state.EX['rs2']
            self.state.EX['ALUResult'], self.state.EX['Zero'] = self.ALU(self.state.EX['rs1'], in2, self.state.EX['aluControlBits'])

            print(f"Forward A: {forwardA}, Forward B: {forwardB}")
            print(f"In1: {self.state.EX['rs1']}, In2: {in2}, ALUResult {self.state.EX['ALUResult']}, Zero: {self.state.EX['Zero']}")

            # branching
            if self.state.EX['Zero'] and self.state.EX['Branch']:
                print("Branching...")
                self.state.IF["PC"] = self.state.EX["PC"] + self.state.EX['Imm']
                self.state.ID['nop'] = True
                if self.state.IF["nop"]:
                    self.nextState.IF["nop"] = self.state.IF["nop"] = False


            # pass forward
            self.nextState.MEM['Wrt_reg_addr']  = self.state.EX['Wrt_reg_addr']
            self.nextState.MEM['MemRead']       = self.state.EX['MemRead']
            self.nextState.MEM['MemWrite']      = self.state.EX['MemWrite']
            self.nextState.MEM['MemToReg']      = self.state.EX['MemToReg']
            self.nextState.MEM['RegWrite']      = self.state.EX['RegWrite']
            self.nextState.MEM['ALUResult']     = self.state.EX['ALUResult']
            self.nextState.MEM['Branch']        = self.state.EX['Branch']
            self.nextState.MEM['rs2']           = self.state.EX['rs2']
            self.nextState.MEM["PC"]            = self.state.EX["PC"]

            self.nextState.MEM["nop"] = False
            # The code below can allow us to reduce the number of cycles, by skipping the MEM and the WB
            # for cases where we don't write back to register and don't use memory. 
            # if self.nextState.MEM['RegWrite'] or self.nextState.MEM['MemRead'] or self.nextState.MEM['MemWrite']:
            #     self.nextState.MEM["nop"] = False
            # else:
            #     self.nextState.MEM["nop"] = True

            print('-'*10)
        else:
            self.clearStage(self.state.EX)
            self.nextState.MEM["nop"] = True
        
        # --------------------- ID stage ---------------------
        if not self.state.ID["nop"]:
            decodedInstr = self.decodeInstr()
            CUOut = self.controlUnit(b2i(decodedInstr['opcode']))
            Reg_rs1, Reg_rs2 = b2i(decodedInstr['rs1']), b2i(decodedInstr['rs2'])

            # Add stall for MEM 1st hazard MemRead
            if (self.state.EX['RegWrite'] and self.state.EX['MemRead'] and self.state.EX['Wrt_reg_addr'] != 0 and 
                (self.state.EX['Wrt_reg_addr'] == Reg_rs1 or (CUOut['ALUSrc'] == 0 and self.state.EX['Wrt_reg_addr'] == Reg_rs2))):
                # clear values in EX
                self.nextState.EX['nop'] = True
                self.state.IF["PC"] = self.state.ID["PC"]
                
            else:
                print("ID Stage <-----")
                print("ID:", self.state.ID)
                print("DECODE_INSTR", decodedInstr)

                self.nextState.EX['Reg_rs1'] = Reg_rs1
                self.nextState.EX['Reg_rs2'] = Reg_rs2
                self.nextState.EX['Wrt_reg_addr'] = b2i(decodedInstr['rd'])
                self.nextState.EX['rs1'] = self.myRF.readRF(Reg_rs1)
                self.nextState.EX['rs2'] = self.myRF.readRF(Reg_rs2)
                self.nextState.EX['ALUOp']      = CUOut['ALUOp']
                self.nextState.EX['ALUSrc']     = CUOut['ALUSrc']
                self.nextState.EX['Branch']     = CUOut['Branch']
                self.nextState.EX['MemRead']    = CUOut['MemRead']
                self.nextState.EX['MemWrite']   = CUOut['MemWrite']
                self.nextState.EX['MemToReg']   = CUOut['MemToReg']
                self.nextState.EX['RegWrite']   = CUOut['RegWrite']  
                self.nextState.EX['Imm'] = self.immGenUnit()
                self.nextState.EX['aluControlBits'] = self.ALUControlUnit(self.nextState.EX['ALUOp'], 
                                                                            b2i(decodedInstr['funct3']), 
                                                                            b2i(decodedInstr['funct7'][1]))
                self.nextState.EX["PC"] = self.state.ID["PC"]
                self.nextState.EX["nop"] = False
            print('-'*10)
        else:
            self.nextState.EX["nop"] = True
            self.clearStage(self.state.ID)
        
        # --------------------- IF stage ---------------------
        if not self.state.IF["nop"]:
            print("IF Stage <-----")
            self.nextState.ID["Instr"] = self.ext_imem.readInstr(self.state.IF["PC"])
            print("IF:", self.state.IF["PC"], self.nextState.ID["Instr"])

            if b2twos(self.nextState.ID["Instr"]) == -1:
                self.nextState.ID["nop"] = True
                self.nextState.IF["nop"] = True
            else:
                self.nextState.IF["PC"] = self.state.IF["PC"] + 4
                self.nextState.ID["nop"] = False

            self.nextState.ID["PC"] = self.state.IF["PC"]
            print('-'*10)
        else:
            self.clearStage(self.state.IF)
            self.nextState.ID["nop"] = True

        print()

        # self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        self.state = copy.deepcopy(self.nextState) #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

    def clearStage(self, stage):
        nopVal = stage['nop']
        for k in stage:
            stage[k] = 0
        stage['nop'] = nopVal


if __name__ == "__main__":
     
    # parse arguments for input file location
    # parser = argparse.ArgumentParser(description='RV32I processor')
    # parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    # args = parser.parse_args()

    testcasesDir = os.path.abspath("input/")
    outputDir = os.path.abspath("output_gk2657/")
    print("Testcases/Input Directory:", testcasesDir)
    print("Ouptut Directory:", outputDir)

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    for tc in sorted(os.listdir(testcasesDir)):
        tcpath = os.path.join(testcasesDir, tc)
        tcOutPath = os.path.join(outputDir, tc)
        imem = InsMem("Imem", tcpath)
        dmem_ss = DataMem("SS", tcpath)
        dmem_fs = DataMem("FS", tcpath)

        if not os.path.isdir(tcOutPath):
            os.mkdir(tcOutPath)
    
        ssCore = SingleStageCore(tcOutPath, imem, dmem_ss)
        fsCore = FiveStageCore(tcOutPath, imem, dmem_fs)
        
        print("Single Stage")
        while(True):
            if not ssCore.halted:
                ssCore.step()
            
            if ssCore.halted:
                break
        
        print("Five Stage")
        while(True):
            if not fsCore.halted:
                fsCore.step()

            if fsCore.halted:
                break
        
        # Write PerformanceMetrics.txt file
        res = "Performance of Single Stage:\n" \
             f"#Cycles -> {ssCore.cycle}\n" \
             f"#Instructions -> {ssCore.instrCount}\n" \
             f"CPI -> {ssCore.cycle / ssCore.instrCount}\n" \
             f"IPC -> {ssCore.instrCount / ssCore.cycle}\n\n" \
             "Performance of Five Stage:\n" \
             f"#Cycles -> {fsCore.cycle}\n" \
             f"#Instructions -> {fsCore.instrCount}\n" \
             f"CPI -> {fsCore.cycle / fsCore.instrCount}\n" \
             f"IPC -> {fsCore.instrCount / fsCore.cycle}\n\n" \
                
        with open(os.path.join(tcOutPath, "PerformanceMetrics_Result.txt"), "w") as file:
            file.write(res)
        
        # dump SS and FS data mem.
        dmem_ss.outputDataMem()
        dmem_fs.outputDataMem()
