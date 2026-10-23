# Đề xuất đặc tả missingness cho benchmark synthetic

Trạng thái: KHÔNG ĐƯỢC CHẤP NHẬN LÀM ĐẶC TẢ TRIỂN KHAI. Người dùng yêu cầu nghiên cứu lại các lựa chọn có công thức bám sát paper, thay vì công thức AI tự tổng hợp. Nội dung bên dưới được giữ để truy vết đề xuất cũ, không dùng làm thiết kế đã chốt. Chưa viết generator.
Lưới missingness, cấu hình dữ liệu và quy mô đã được xác nhận trong hội thoại.

## 1. Ký hiệu và phạm vi thông tin

- i: thực thể; t=1..120: thời điểm; j=1..12: component.
- X_itj: counts đầy đủ; p_itj: tỷ lệ tiềm ẩn; N_it: tổng counts; a_i: entity_trait_01.
- H_itj=1: ô được che. Chỉ X bị che; N, thời gian, entity ID và a luôn quan sát.
- M=E*T*D là số ô component, bao gồm cả zero. Phần dư và metadata không thuộc mẫu số missing rate.
- p, nhãn zero thật/zero lấy mẫu và phần dư không được đưa vào conditioning của baseline.
- Mỗi kịch bản tạo mask riêng; không trộn các loại mask trong cùng một cấu hình trừ khi có thiết kế khác được phê duyệt.

## 2. Đơn vị gây thiếu: phân hoạch không chồng ô

Chọn một phân hoạch ngẫu nhiên độc lập với X,p,N,a trước khi tính xác suất. Các đơn vị U tạo thành phân hoạch toàn bộ M ô, nên mỗi ô thuộc đúng một đơn vị.

| Dạng | Một đơn vị U | Số ô w_U |
|---|---|---:|
| Rải rác | Một ô (i,t,j) | 1 |
| Khối một phần | Một thực thể, L thời điểm liên tiếp, nhóm q component | L*q |
| Khối toàn bộ | Một thực thể, L thời điểm liên tiếp, 12 component | L*12 |
| Cả component/thực thể | Một cặp (i,j), toàn bộ 120 thời điểm | 120 |

Khối: chia 120 thời điểm thành cửa sổ L không chồng thời gian; trong mỗi cửa sổ của mỗi thực thể, hoán vị ngẫu nhiên 12 component rồi chia thành nhóm q. L=6/12/24 đều chia hết 120; q=1/3/6/12 đều chia hết 12. Không cần khối biên ngắn hoặc cắt ô để đủ kích thước.

Các đơn vị không chồng ô nhưng có thể kề nhau hoặc cùng cửa sổ trên các nhóm component khác nhau. Hệ quả cần được người dùng chấp thuận:
- L là độ dài khối cơ sở, không bảo đảm đoạn thiếu liên tục cuối cùng dài đúng L. Các khối kề nhau có thể nối thành đoạn dài hơn.
- q là số component trong mỗi khối cơ sở, không phải số component tối đa bị thiếu tại một thời điểm. Nhiều nhóm trong cùng cửa sổ có thể cùng được chọn, thậm chí toàn bộ 12 component.
- Giới hạn tối đa 3/12 component thiếu ở mọi thời điểm sẽ làm rate không vượt .25; không tương thích với rate .50. Muốn khống chế số component hoặc độ dài đoạn thực tế chính xác phải đổi thiết kế/miền khả thi và hiệu chỉnh lại xác suất.
- Báo cả L,q cấu hình lẫn phân phối run lengths và số component thiếu thực tế; không gọi hai đại lượng này giống nhau.

Đây là thiết kế khối mở rộng cho benchmark này, không tuyên bố sao chép nguyên thuật toán của paper.

## 3. Chuẩn hóa

Với giá trị v_U trên đơn vị, đặt Z_w(v_U)=(v_U-mu_w)/sd_w, trong đó mu_w và sd_w tính với trọng số w_U. Dùng population SD (ddof=0). Nếu sd_w=0 thì trả điểm 0 và log cờ score_degenerate; không chia cho epsilon để tạo tín hiệu giả.

Chuẩn hóa log N ở cấp ô cũng dùng toàn bộ entity-time/component weights; a được chuẩn hóa ở cấp thực thể. N phải nguyên dương và hữu hạn. a phải đầy đủ và hữu hạn.

Các đại lượng này thuộc mask generator. Không dùng scaler mask cho model; model fit encoder/scaler riêng chỉ trên train. Với MAR chỉ sử dụng N,a quan sát; với MNAR p được dùng có chủ đích để tạo mask, không dùng fit/tune baseline.

## 4. Điểm gây thiếu và xác suất

sigmoid(z)=1/(1+exp(-z)). Với mỗi đơn vị U:

    pi_U = sigmoid(alpha + beta*S_U)
    B_U ~ Bernoulli(pi_U), độc lập giữa đơn vị khi đã biết dữ liệu và phân hoạch
    H_c = B_U cho mọi ô c thuộc U

### MCAR

    pi_U = r

Phân hoạch và random draw độc lập giá trị; các ô trong cùng khối có missing indicators phụ thuộc nhau do cùng B_U, vẫn không phụ thuộc giá trị dữ liệu.

### MAR

Đặt zN_it=Z(log N_it), za_i=Z(a_i). Đề xuất:

    v_U = -wN * mean_U(zN_it) + wa * za_i
    S_U = Z_w(v_U)

Mặc định wN=wa=1, beta=1. N thấp, trait cao làm tăng xác suất thiếu, giữ các yếu tố khác cố định và dưới cùng bộ chuẩn hóa/calibration. wN/wa điều khiển tương quan đóng góp; beta điều khiển độ mạnh chung. Có thể thay riêng các trọng số nhưng phải log cả hai và bộ chuẩn hóa vì thay trọng số cũng đổi SD tổng.

Sensitivity đã chốt beta=.5/2 thay độ mạnh chung, giữ tỷ trọng hai yếu tố. beta=0 trở thành MCAR ở cấp đơn vị. Không đưa p,X hay phần dư vào MAR score hoặc bước quyết định phân hoạch.

### MNAR điều khiển bởi latent p

    epsilon = 1e-6
    rarity_itj = -log(p_itj + epsilon)
    v_U = mean_U(rarity_itj)
    S_U = Z_w(v_U)

Với ô riêng, p nhỏ hơn tạo điểm cao hơn. Với khối, đó là độ hiếm trung bình trên toàn khối theo thang log, không phải bảo đảm xác suất thiếu của từng ô chỉ phụ thuộc riêng p của ô đó.

p=0 có rarity hữu hạn lớn nhất -log(1e-6), không bị gán pi=1 và không bị loại khỏi masking. epsilon chỉ phục vụ score, tuyệt đối không sửa p, X, nhãn zero, baseline input hoặc evaluation truth. Sensitivity epsilon ngoài lưới chính nếu cần; không lén thay epsilon theo dữ liệu hay baseline.

Đây là latent-driven missingness. Không khẳng định có thể xác định cơ chế MNAR từ dữ liệu quan sát; formal classification cần xét mức thông tin thực sự được baseline quan sát và các phụ thuộc của generator. Trong thiết kế p không được cung cấp và không được quyết định hoàn toàn bởi biến quan sát.

## 5. Hiệu chỉnh tỷ lệ kỳ vọng

Đối với MAR/MNAR, tìm một alpha chung cho toàn bộ panel/cấu hình bằng bisection:

    f(alpha) = sum_U w_U*sigmoid(alpha+beta*S_U) / M - r = 0

f liên tục, đơn điệu; r trong (0,1). Mở rộng bracket đến khi bao được nghiệm; dùng expit ổn định số, kiểm residual tuyệt đối <=1e-8. Không chọn alpha từ accuracy của baseline.

- Có một alpha cho cả train/validation/test, không hiệu chỉnh riêng từng thực thể hoặc ép từng thực thể có cùng tỷ lệ.
- Quy tắc tạo mask được áp dụng trước khi fit baseline. Toàn panel được dùng trong bước tạo missingness tổng hợp; đây không phải fit model lên test. MAR calibration chỉ dùng thông tin luôn quan sát. MNAR calibration sử dụng p đúng mục đích sinh mask.
- Nếu yêu cầu không dùng ngay cả phân phối truth test cho mask calibration, cần thay bằng calibration panel độc lập và chấp nhận sai lệch kỳ vọng trên evaluation panel; đây là protocol khác, chưa tự chọn.
- Không điều chỉnh slope beta để đạt rate: chỉ đổi alpha.
- Mỗi profile/pattern/rate/beta có alpha được log riêng; công thức và trọng số giống nhau giữa split, alpha không tính lại theo split.

Rate thực tế:

    r_hat = sum_U w_U*B_U / M
    SE(r_hat | panel,partition) = sqrt(sum_U w_U^2*pi_U*(1-pi_U))/M

Đề xuất bảo đảm rate kỳ vọng r, không ép exact count trong mỗi mask realization. Báo r, r_hat, SE và tỷ lệ từng split/entity/stratum. Rate của strata là missing fraction trong từng strata, phân biệt với tỷ trọng strata trong tổng số hidden cells.

Giữ draw đầu tiên với seed đã đăng ký, không retry đến khi có rate hoặc score mong muốn. Không cắt khối, thêm/bớt ô hoặc force ít nhất một observed component sau draw. Nếu không còn donor/complete rows, báo trạng thái và fallback theo protocol đã đăng ký; không sửa mask riêng cho model.

## 6. Cùng điều kiện giữa baseline và kiểm tra

- Mỗi dataset/pattern/rate/beta/seed sinh một mask chung cho tất cả baseline; lưu hash và seed độc lập cho partition/draw.
- Cùng grid không có nghĩa cùng missing-cell IDs giữa easy/medium/hard: MAR/MNAR thay theo N,p của profile. Có thể dùng common uniform draws khi so rate nhưng không thay rate selection theo baseline performance.
- Với phân hoạch/score cố định và cùng uniforms, tăng r chỉ tăng alpha, cho phép masks nested theo rate.
- Không chỉ đo correlation mask-p hay mask-N để chứng minh causal mechanism. Kiểm tra công thức bằng perturbation giữ calibration cố định, nhãn miền thông tin và conditional probabilities.
- Kiểm tra partition phủ đúng M ô một lần; score/probabilities hữu hạn; MCAR không phụ thuộc truth; MAR không truy cập p/X; MNAR không sửa p=0; toàn khối che cùng nhau.
- Audit train-only model fitting, conditioning không chứa labels/residual/private p, cùng mask giữa models, và báo support của ba strata chấm điểm.

## 7. Những điểm cần xác nhận trước khi code

1. Rate là kỳ vọng, draw thực tế có dao động; không ép đúng số ô từng lần.
2. L,q là kích thước khối cơ sở; đoạn liên tục/số component thiếu thực tế có thể lớn hơn do các khối được chọn đồng thời/kề nhau.
3. Mask-only epsilon=1e-6; MAR weights 1:1; score khối là trung bình rồi chuẩn hóa.
4. Calibration trên toàn panel với một alpha chung; không fit model trên truth test.

Nguồn phương pháp tham khảo: Schouten, Lugtig & Vink (2018), Generating missing values for simulation purposes: a multivariate amputation procedure, https://doi.org/10.1080/00949655.2018.1491577 . Công thức cụ thể, latent-p scores và phân hoạch khối ở đây là đề xuất của AI cho yêu cầu người dùng.
