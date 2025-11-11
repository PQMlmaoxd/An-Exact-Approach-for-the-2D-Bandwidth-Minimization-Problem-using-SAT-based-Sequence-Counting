# Hybrid Distance Encoder - Usage Guide

## Tổng quan

File `distance_encoder_hybrid.py` cung cấp một cách tiếp cận **hybrid** để encode distance constraints, cho phép bạn **thay thế dần dần** các T variables bằng mutual exclusion clauses.

## Mục đích

Cho phép so sánh performance giữa các mức độ thay thế khác nhau tại một K cụ thể:
- **Standard encoding**: Tất cả T variables với activation clauses (chậm nhưng linh hoạt)
- **Cutoff encoding**: Chỉ T variables đến UB với mutual exclusions cho phần còn lại (nhanh, tối ưu)
- **Hybrid encoding**: Kết hợp cả hai - cho phép thay thế từng bước

## Cách sử dụng với `custom_k_bandwidth_solver.py`

### 1. Standard encoding (baseline)
```bash
python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4 --method=standard
```
- Tạo tất cả T variables từ T_1 đến T_{n-1}
- Sử dụng activation clauses cho tất cả
- **Use case**: Baseline để so sánh

### 2. Cutoff encoding (optimized)
```bash
python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4 --method=cutoff
```
- Tạo T variables chỉ đến T_UB (theoretical upper bound)
- Sử dụng mutual exclusions cho distances > UB
- **Use case**: Production, performance tối ưu

### 3. Hybrid encoding (incremental comparison)

#### Replace 1 level (T_{UB+1} only)
```bash
python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4 --method=hybrid --replacements=1
```
- Giữ: T_1 đến T_UB với activation clauses
- Thay: T_{UB+1} với mutual exclusions
- **Use case**: Test ảnh hưởng của việc thay thế 1 level

#### Replace 5 levels
```bash
python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4 --method=hybrid --replacements=5
```
- Giữ: T_1 đến T_UB với activation clauses
- Thay: T_{UB+1} đến T_{UB+5} với mutual exclusions
- **Use case**: Test ảnh hưởng của việc thay thế nhiều levels

#### Full replacement (equivalent to cutoff)
```bash
python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4 --method=hybrid --replacements=100
```
- Giữ: T_1 đến T_UB với activation clauses
- Thay: T_{UB+1} đến T_{n-1} với mutual exclusions
- **Use case**: Verify equivalence với cutoff encoding
- **Note**: Tự động giới hạn tại max_replacements = n-1-UB

## Performance Comparison Tool

Để so sánh tất cả các phương pháp cùng lúc:

```bash
python test_hybrid_performance.py bcsstk01.mtx 4 --solver=cadical195
```

Tool này sẽ test:
1. Standard encoding
2. Cutoff encoding
3. Hybrid với 1 replacement
4. Hybrid với 5 replacements (nếu có thể)
5. Hybrid với full replacement

Và hiển thị bảng so sánh về:
- Số variables
- Số clauses
- Thời gian solve
- Performance ratios

## Các ví dụ cụ thể

### Ví dụ 1: So sánh trên instance nhỏ
```bash
# Test với các phương pháp khác nhau
python custom_k_bandwidth_solver.py jgl009.mtx cadical195 10 --method=standard
python custom_k_bandwidth_solver.py jgl009.mtx cadical195 10 --method=cutoff
python custom_k_bandwidth_solver.py jgl009.mtx cadical195 10 --method=hybrid --replacements=1
python custom_k_bandwidth_solver.py jgl009.mtx cadical195 10 --method=hybrid --replacements=100

# Hoặc chạy tất cả cùng lúc
python test_hybrid_performance.py jgl009.mtx 10 --solver=cadical195
```

### Ví dụ 2: So sánh trên instance lớn
```bash
python test_hybrid_performance.py ash85.mtx 25 --solver=cadical195
```

### Ví dụ 3: Tìm mức replacement tối ưu
```bash
# Test từng mức để tìm sweet spot
for repl in 0 1 2 3 5 10 20 50 100; do
    echo "Testing replacements=$repl"
    python custom_k_bandwidth_solver.py ck104.mtx cadical195 15 --method=hybrid --replacements=$repl
done
```

## Kỳ vọng về Performance

### Variables & Clauses
- **Standard**: Nhiều nhất (T_1 to T_{n-1})
- **Hybrid**: Giảm dần theo số replacements
- **Cutoff**: Ít nhất (T_1 to T_UB)

### Solve Time
Thường theo thứ tự (nhanh → chậm):
1. Cutoff (hoặc Hybrid full replacement)
2. Hybrid với một số replacements
3. Standard

Tuy nhiên, điều này **phụ thuộc vào instance** và **K value**.

## Important Notes

### 1. T Variable Semantics trong Hybrid Mode

⚠️ **QUAN TRỌNG**: Trong hybrid mode với replacements > 0, các T variables được giữ lại (T_1 đến T_UB) **không bắt buộc** phải được set đúng giá trị trong model SAT.

**Ví dụ**: 
- Distance thực tế = 4, UB = 3, replacements = 1
- Hybrid encoder tạo: T_1, T_2, T_3 (với activation) + mutual exclusions cho distance ≥ 5
- SAT solver có thể trả về: T_1=T_2=T_3=False (mặc dù distance=4 ≥ 1,2,3)

**Tại sao?**
- Activation clauses chỉ là "sufficient direction": (U_i ∧ V_k) → T_d
- Không có "necessary direction": T_d → ∨(U_i ∧ V_k)
- Solver không bắt buộc phải set T variables nếu không cần thiết cho các constraints khác

**Có vấn đề không?**
- **KHÔNG** - Trong context của bandwidth constraints!
- Bandwidth constraint chỉ check `¬T_{K+1}` để enforce distance ≤ K
- Miễn là mutual exclusions ngăn distance > replaced threshold, mọi thứ đều đúng
- T variables không được set chính xác không ảnh hưởng đến correctness của solution

### 2. Khi nào nên dùng phương pháp nào?

**Standard encoding:**
- Khi cần linh hoạt nhất
- Khi K value rất gần n-1
- Cho research/debugging

**Cutoff encoding:**
- **RECOMMENDED** cho production
- Khi K value gần theoretical UB
- Khi cần performance tốt nhất
- Khi graph có theoretical bound chặt

**Hybrid encoding:**
- **Cho research** để hiểu ảnh hưởng của replacements
- **Performance comparison** giữa các mức độ replacement
- **Validate** rằng full replacement = cutoff
- **Không khuyến khích** cho production use

### 3. Verification của Correctness

Để verify rằng solution đúng, hãy check:
1. ✅ Tất cả edges thỏa mãn distance ≤ K (extract từ model)
2. ✅ SAT/UNSAT result khớp giữa các phương pháp
3. ❌ KHÔNG cần check T variable values trong hybrid mode

## Kết luận

Hybrid encoder là một **tool để nghiên cứu và so sánh**, không phải để production use. Nếu bạn cần:
- **Performance**: Dùng cutoff encoding
- **Flexibility**: Dùng standard encoding  
- **Research**: Dùng hybrid để hiểu trade-offs

Đối với production code trong `custom_k_bandwidth_solver.py`, khuyến nghị sử dụng `--method=cutoff`.
