#!/usr/bin/env python3
"""
FP16 x INT4 CIM计算模拟器 - 完整反量化流程
支持显式的scale和zero_point参数
"""
import numpy as np

class FP16_INT4_CIM_Simulator:
    """FP16 x INT4 在RRAM CIM阵列上的计算模拟器"""
    
    def __init__(self):
        self.verbose = True
        
    def float_to_fp16_components(self, value):
        """将浮点数转换为FP16的组件（符号、指数、尾数）"""
        fp16_val = np.float16(value)
        bits = np.frombuffer(fp16_val.tobytes(), dtype=np.uint16)[0]
        
        sign = (bits >> 15) & 0x1
        exponent = (bits >> 10) & 0x1F
        mantissa = bits & 0x3FF
        
        return sign, exponent, mantissa, fp16_val
    
    def int4_to_binary(self, value):
        """将INT4转换为二进制（只做RRAM映射，不处理零点）"""
        if value < -8 or value > 7:
            raise ValueError(f"INT4值超出范围: {value} (应在-8到7之间)")
        
        # 只做RRAM映射：加8映射到0-15范围
        mapped_value = value + 8
        
        # 拆分为MSB和LSB（各2位）
        msb = (mapped_value >> 2) & 0b11
        lsb = mapped_value & 0b11
        
        return msb, lsb, mapped_value
    
    def align_mantissas(self, a1_e, a1_m, a2_e, a2_m):
        """对齐两个尾数"""
        max_e = max(a1_e, a2_e)
        
        # 加入隐含的1，转换为完整尾数 (1.mantissa)
        a1_full = (1 << 10) | a1_m
        a2_full = (1 << 10) | a2_m
        
        # 对齐
        shift1 = max_e - a1_e
        shift2 = max_e - a2_e
        
        a1_m_aligned = a1_full >> shift1 if shift1 > 0 else a1_full
        a2_m_aligned = a2_full >> shift2 if shift2 > 0 else a2_full
        
        return max_e, a1_m_aligned, a2_m_aligned, shift1, shift2
    
    def cim_compute(self, a1_s, a1_m, a2_s, a2_m, b1_msb, b1_lsb, b2_msb, b2_lsb):
        """在2x2 RRAM CIM阵列上执行计算"""
        # RRAM阵列
        rram_array = [[b1_msb, b1_lsb],
                      [b2_msb, b2_lsb]]
        
        # 累加器
        accumulator = 0
        
        # 计算需要的周期数（每周期处理2位）
        max_bits = max(int(a1_m).bit_length(), int(a2_m).bit_length())
        num_cycles = (max_bits + 1) // 2
        
        if self.verbose:
            print(f"\n  CIM计算详细过程 ({num_cycles}个周期):")
        
        # 处理所有周期
        for cycle in range(num_cycles):
            # 从a1_m和a2_m提取当前周期的2位（从低到高）
            a1_bits = (a1_m >> (cycle * 2)) & 0b11
            a2_bits = (a2_m >> (cycle * 2)) & 0b11
            
            cycle_contribution = 0
            
            if self.verbose:  # 显示所有周期
                print(f"    周期{cycle}: a1第{cycle*2}-{cycle*2+1}位={a1_bits:02b}, a2第{cycle*2}-{cycle*2+1}位={a2_bits:02b}")
            
            # 计算当前周期的部分和
            for col in range(2):
                # 根据符号位决定加减
                if a1_s == 0 and a2_s == 0:
                    partial_sum = a1_bits * rram_array[0][col] + a2_bits * rram_array[1][col]
                elif a1_s == 0 and a2_s == 1:
                    partial_sum = a1_bits * rram_array[0][col] - a2_bits * rram_array[1][col]
                elif a1_s == 1 and a2_s == 0:
                    partial_sum = -a1_bits * rram_array[0][col] + a2_bits * rram_array[1][col]
                else:
                    partial_sum = -a1_bits * rram_array[0][col] - a2_bits * rram_array[1][col]
                
                # 移位并累加 (shift = 2*i + 2*(1-j), i=cycle, j=col)
                shift = 2 * cycle + 2 * (1 - col)
                contribution = partial_sum << shift
                accumulator += contribution
                cycle_contribution += contribution
                
                if self.verbose:
                    # 显示详细计算
                    if a1_s == 0 and a2_s == 1:
                        op_str = f"{a1_bits}*{rram_array[0][col]} - {a2_bits}*{rram_array[1][col]}"
                    elif a1_s == 0 and a2_s == 0:
                        op_str = f"{a1_bits}*{rram_array[0][col]} + {a2_bits}*{rram_array[1][col]}"
                    else:
                        op_str = f"复杂符号运算"
                    
                    print(f"      列{col}: {op_str} = {partial_sum}, 左移{shift}位 = {contribution}")
            
            if self.verbose:
                print(f"      本周期贡献: {cycle_contribution}, 累计: {accumulator}")
        
        return accumulator
    
    def compute_full_dequantization(self, a1, a2, b1, b2, scale=1.0, zero_point=0):
        """执行完整的FP16 x INT4反量化计算：(a1*b1 + a2*b2) * scale - A_Sum * zero_point"""
        print(f"\n{'='*80}")
        print(f"完整反量化计算")
        print(f"输入: a1={a1}, a2={a2}, b1={b1}, b2={b2}")
        print(f"量化参数: scale={scale}, zero_point={zero_point}")
        print(f"计算: ((a1*b1 + a2*b2) * scale) - (a1+a2) * zero_point")
        
        # 1. FP16转换和A_Sum计算
        a1_s, a1_e, a1_m, a1_fp16 = self.float_to_fp16_components(a1)
        a2_s, a2_e, a2_m, a2_fp16 = self.float_to_fp16_components(a2)
        
        # A_Sum: 完整FP16激活和 (Digital Chiplet累加)
        A_Sum = float(a1_fp16) + float(a2_fp16)
        
        print(f"\n1. FP16转换和A_Sum计算:")
        print(f"   a1: {a1} -> {a1_fp16} (s={a1_s}, e={a1_e}, m={a1_m:010b})")
        print(f"   a2: {a2} -> {a2_fp16} (s={a2_s}, e={a2_e}, m={a2_m:010b})")
        print(f"   A_Sum = {float(a1_fp16)} + {float(a2_fp16)} = {A_Sum}")
        
        # 2. 尾数对齐得到标定指数Max_E
        max_e, a1_m_aligned, a2_m_aligned, shift1, shift2 = self.align_mantissas(a1_e, a1_m, a2_e, a2_m)
        
        # P_Sum: 移位截断后的带符号尾数之和
        # 注意：这里需要正确处理符号位
        a1_signed = a1_m_aligned if a1_s == 0 else -int(a1_m_aligned)
        a2_signed = a2_m_aligned if a2_s == 0 else -int(a2_m_aligned)
        P_Sum = a1_signed + a2_signed
        
        print(f"\n2. 尾数对齐和P_Sum计算 (Max_E={max_e}):")
        print(f"   a1: {(1<<10)|a1_m} >> {shift1} = {a1_m_aligned}")
        print(f"   a2: {(1<<10)|a2_m} >> {shift2} = {a2_m_aligned}")
        print(f"   a1_signed = {a1_signed}, a2_signed = {a2_signed}")
        print(f"   P_Sum = {a1_signed} + {a2_signed} = {P_Sum}")
        
        # 3. INT4转换（只做RRAM映射，+8）
        b1_msb, b1_lsb, b1_mapped = self.int4_to_binary(b1)
        b2_msb, b2_lsb, b2_mapped = self.int4_to_binary(b2)
        
        print(f"\n3. INT4转换 (只做RRAM映射 +8):")
        print(f"   b1: {b1} + 8 -> {b1_mapped:04b} (msb={b1_msb:02b}, lsb={b1_lsb:02b})")
        print(f"   b2: {b2} + 8 -> {b2_mapped:04b} (msb={b2_msb:02b}, lsb={b2_lsb:02b})")
        
        # 4. RRAM阵列显示
        print(f"\n4. RRAM阵列:")
        print(f"     列0(MSB)  列1(LSB)")
        print(f"   行0:  {b1_msb:02b}       {b1_lsb:02b}")
        print(f"   行1:  {b2_msb:02b}       {b2_lsb:02b}")
        
        # 5. CIM计算得到I_Sum
        print(f"\n5. CIM计算:")
        I_Sum = self.cim_compute(a1_s, a1_m_aligned, a2_s, a2_m_aligned, 
                                 b1_msb, b1_lsb, b2_msb, b2_lsb)
        
        # 6. 计算O_m (输出尾数位): I_Sum - P_Sum * 8
        P_Sum_offset = int(P_Sum) * 8
        O_m = int(I_Sum) - P_Sum_offset
        
        print(f"\n6. 输出尾数计算:")
        print(f"   I_Sum = {I_Sum}")
        print(f"   P_Sum = {P_Sum}")
        print(f"   O_m = I_Sum - P_Sum * 8 = {I_Sum} - {P_Sum_offset} = {O_m}")
        
        # 7. 计算输出指数 O_e = Max_E - 10  
        O_e = int(max_e) - 10
        
        # 8. 转换为浮点数 O
        # 注意：需要正确计算FP16的指数偏移
        # FP16指数偏移是15，所以实际指数 = O_e - 15
        actual_exponent = int(O_e) - 15
        
        # 避免指数溢出，直接使用numpy的float16精度
        if actual_exponent > 15:  # FP16最大指数约为15
            print(f"   ⚠️ 警告: 指数{actual_exponent}超出FP16范围，结果可能溢出")
            base_scale_factor = float('inf')
            O = float('inf') if O_m > 0 else float('-inf')
        elif actual_exponent < -24:  # FP16最小指数约为-24
            print(f"   ⚠️ 警告: 指数{actual_exponent}低于FP16范围，结果趋向于0")
            base_scale_factor = 0.0
            O = 0.0
        else:
            base_scale_factor = 2.0 ** actual_exponent
            O = O_m * base_scale_factor
        
        print(f"\n7. 转换为浮点数 O:")
        print(f"   O_e = Max_E - 10 = {max_e} - 10 = {O_e}")
        print(f"   实际指数 = O_e - 15 = {O_e} - 15 = {actual_exponent}")
        print(f"   基础缩放因子: 2^{actual_exponent} = {base_scale_factor}")
        print(f"   O = {O_m} × {base_scale_factor} = {O}")
        
        # 9. 完整反量化: result = O × scale - A_Sum × zero_point × scale
        # 按照PyTorch反量化公式的等价形式
        scaled_result = O * scale
        zero_point_correction = A_Sum * zero_point * scale  # 注意：零点也要乘scale
        final_result = scaled_result - zero_point_correction
        
        print(f"\n8. 完整反量化计算:")
        print(f"   O × scale = {O} × {scale} = {scaled_result}")
        print(f"   A_Sum × zero_point × scale = {A_Sum} × {zero_point} × {scale} = {zero_point_correction}")
        print(f"   最终结果 = {scaled_result} - {zero_point_correction} = {final_result}")
        
        # 10. 验证 - 按照标准PyTorch反量化公式
        b1_dequantized = (b1 - zero_point) * scale
        b2_dequantized = (b2 - zero_point) * scale
        expected = float(a1_fp16) * b1_dequantized + float(a2_fp16) * b2_dequantized
        
        error = abs(final_result - expected)
        error_percent = (error / abs(expected) * 100) if expected != 0 else 0
        
        print(f"\n9. 验证 (标准PyTorch反量化):")
        print(f"   b1_dequant = ({b1} - {zero_point}) × {scale} = {b1_dequantized}")
        print(f"   b2_dequant = ({b2} - {zero_point}) × {scale} = {b2_dequantized}")
        print(f"   预期结果: {float(a1_fp16)} × {b1_dequantized} + {float(a2_fp16)} × {b2_dequantized} = {expected}")
        print(f"   SA结果: {final_result}")
        print(f"   误差: {error:.6f} ({error_percent:.4f}%)")
        
        return {
            'sa_result': final_result,
            'expected': expected,
            'error': error,
            'error_percent': error_percent,
            'O': O,
            'A_Sum': A_Sum,
            'scale': scale,
            'zero_point': zero_point
        }
    
    def compute(self, a1, a2, b1, b2):
        """原始计算方法（保持兼容性）"""
        return self.compute_full_dequantization(a1, a2, b1, b2, scale=1.0, zero_point=0)

def main():
    """主函数 - 交互式界面"""
    print("FP16 x INT4 CIM 完整反量化模拟器")
    print("="*80)
    print("计算: ((a1×b1 + a2×b2) × scale) - (a1+a2) × zero_point")
    print("- a1, a2: FP16浮点数")
    print("- b1, b2: INT4整数（-8到7）")
    print("- scale: 反量化缩放因子")
    print("- zero_point: 反量化零点偏移")
    print("="*80)
    
    simulator = FP16_INT4_CIM_Simulator()
    
    while True:
        print("\n选择输入方式:")
        print("1. 使用示例值（带量化参数）")
        print("2. 手动输入完整参数")
        print("3. 简单测试（默认scale=1, zero_point=0）")
        print("4. 退出")
        
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            # 使用示例值
            print("\n选择示例:")
            examples = [
                # (a1, a2, b1, b2, scale, zero_point)
                (100.0, 50.0, 3, -2, 0.1, 0),      # 基础测试
                (10.0, 20.0, 1, -1, 0.5, 1),       # 有零点偏移
                (1.5, -2.5, 7, -8, 0.25, -2),      # 极值测试
                (0.125, 0.25, 4, 2, 2.0, 3),       # 小数激活
            ]
            for i, (a1, a2, b1, b2, scale, zp) in enumerate(examples):
                print(f"{i+1}. a1={a1}, a2={a2}, b1={b1}, b2={b2}, scale={scale}, zp={zp}")
            
            try:
                idx = int(input("\n选择示例编号: ")) - 1
                if 0 <= idx < len(examples):
                    a1, a2, b1, b2, scale, zero_point = examples[idx]
                else:
                    print("无效的选择")
                    continue
            except ValueError:
                print("输入错误")
                continue
                
        elif choice == '2':
            # 手动输入完整参数
            try:
                a1 = float(input("输入 a1 (FP16浮点数): "))
                a2 = float(input("输入 a2 (FP16浮点数): "))
                b1 = int(input("输入 b1 (INT4, -8到7): "))
                b2 = int(input("输入 b2 (INT4, -8到7): "))
                scale = float(input("输入 scale (反量化缩放因子): "))
                zero_point = int(input("输入 zero_point (反量化零点): "))
                
                if b1 < -8 or b1 > 7 or b2 < -8 or b2 > 7:
                    print("INT4值超出范围！")
                    continue
            except ValueError:
                print("输入格式错误！")
                continue
                
        elif choice == '3':
            # 简单测试
            try:
                a1 = float(input("输入 a1 (FP16浮点数): "))
                a2 = float(input("输入 a2 (FP16浮点数): "))
                b1 = int(input("输入 b1 (INT4, -8到7): "))
                b2 = int(input("输入 b2 (INT4, -8到7): "))
                scale = 1.0
                zero_point = 0
                
                if b1 < -8 or b1 > 7 or b2 < -8 or b2 > 7:
                    print("INT4值超出范围！")
                    continue
            except ValueError:
                print("输入格式错误！")
                continue
                
        elif choice == '4':
            print("\n退出程序")
            break
        else:
            print("无效的选择")
            continue
        
        # 执行计算
        result = simulator.compute_full_dequantization(a1, a2, b1, b2, scale, zero_point)
        
        # 询问是否继续
        cont = input("\n是否继续？(y/n): ").strip().lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()