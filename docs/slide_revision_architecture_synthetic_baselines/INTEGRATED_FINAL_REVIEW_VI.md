# Rà soát tích hợp cuối — finding → thay đổi → evidence

**Kết quả:**32trang (30nội dung+2thamkhảo), XeLaTeX2passes exit0, desktopcompiler success,32renders150dpi.67citation links→10mục ở31/32 và4DOI URIs đã kiểm trong PDF thực. Không có lỗi còn mở trong scope tích hợp.

## Findings chính

|Finding|Thay đổi|Evidence cuối|
|---|---|---|
|Footersource nhỏ/dài trên29trang|Bỏ nguồn/filename ởmain, citation số[1]–[10] gắn caption; ref31/32|PDFpages1–32; pdf_link_audit.json; citation_map.json|
|Nguồn/tácgiả/phânvai paper-code chưa đồngnhất|IEEE-style4papers,6nhómprojectsources; Tsagrisauthororder sửa theoarXiv|Trang31–32; companion; bibliographyprimaryURLs|
|Trang2/6labels sát/đè node; runtime4quádày|Tách2cột, giữvídụtrongcanvas,4runtime stages|renders02,04,06; source.diff|
|Predictionarrow8cắt vùngcaption, generator9nhãnchồng|Route mépphải; r táchkhỏinameds; khoảngtrốngpanel|renders08–09 đãmởlạisaucompilecuối|
|Profileplot10lặpyticks vànguồndài|Sharedaxes, hidenhãnlặp; sample selection ởcompanion|render10;120rows×3profilesarrayaudit|
|Run-scope MICoDA dễđọcnhầm|Interceptchỉsynthetic, D_c tổngquát; COVID5/5foldsđạtcap chưađạttol|renders17,19,20; code/manifests/report|
|Bỏfooter cóthểlàmmấtCI/unit/limits|Chuyển phầnýnghĩasangbody25–29, giữpairedmacrokhácpooled|renders25–29; numerical_audit.json|
|Pairedplotmốc0chưanổi|Thêmđường0nétđứt, giữđúngestimate/CI|render27; CItriplesaudit; builder + manifest|
|Bảngfilephụlục32dàihơntrang|Rútrole, cộtfilenamesraggedright, bỏdòngthừakhônggiảmfont|render32; releasecompilelog|

## Rà từng trang

Từngảnh đãmởở945×532px,150dpi. Saucompilecuối,30ảnhngoài8–9byte-identicalvớiảnhđãxem;8–9đãmởriênglạisaufix. Contactsheet32trangđãxem. Không dùnglogno-overfullthayvisualQA.

|Trang|Finding/kiểm và thay đổi|Evidence render|Verdict|
|---|---|---|---|
|1|Giữ bìa, ví dụ mask và ba baseline; thêm citation phương pháp.|[PNG01](../../presentation/implementation_report/qa_integrated/final/slide-01.png)|PASS|
|2|Bỏ hai box đè nhau, tách hai cột; example A20/B?/C10,N100 không chạm nguồn đánh giá.|[PNG02](../../presentation/implementation_report/qa_integrated/final/slide-02.png)|PASS|
|3|Tách tầngmodule với runtime, hạ khoảng cách hàng labels/evaluator, rút caption.|[PNG03](../../presentation/implementation_report/qa_integrated/final/slide-03.png)|PASS|
|4|12ô thành4stages, labels và node tách; ghi khác biệt refitCOVID/synthetic.|[PNG04](../../presentation/implementation_report/qa_integrated/final/slide-04.png)|PASS|
|5|Other là phần ngoài named; làm rõ output A20/B25/C10/Other45 là minh họa.|[PNG05](../../presentation/implementation_report/qa_integrated/final/slide-05.png)|PASS|
|6|Bỏ ví dụ ngoài canvas, đặt cấu hình trong phạm vi; registrytext không chạm node.|[PNG06](../../presentation/implementation_report/qa_integrated/final/slide-06.png)|PASS|
|7|Giữ capability/CLI/API/spec; giới hạn MCAR/MAR rõ.|[PNG07](../../presentation/implementation_report/qa_integrated/final/slide-07.png)|PASS|
|8|Đổi đường prediction→evaluator sang mép phải, không cắt dòng entity/time.|[PNG08](../../presentation/implementation_report/qa_integrated/final/slide-08.png)|PASS|
|9|Bỏ nhãn p/r đè nhau; rời lời giải thích khỏi wave và formula; nguồn generator[6].|[PNG09](../../presentation/implementation_report/qa_integrated/final/slide-09.png)|PASS|
|10|Chỉ ylabel/ticks chung, bỏ lặp ylabels ở panel2/3; cùngsample/presets.|[PNG10](../../presentation/implementation_report/qa_integrated/final/slide-10.png)|PASS|
|11|Đổi “ba trạng thái zero” thành “ba loại ô theo truth”; mask khácstrata.|[PNG11](../../presentation/implementation_report/qa_integrated/final/slide-11.png)|PASS|
|12|Giữ entitysplit36/12/12 và evaluatorboundary; refit48 đúngsavedrun.|[PNG12](../../presentation/implementation_report/qa_integrated/final/slide-12.png)|PASS|
|13|Định nghĩa ratio→pp; n0null, mộtseed và cells không độc lập.|[PNG13](../../presentation/implementation_report/qa_integrated/final/slide-13.png)|PASS|
|14|Giữ fit-donorstate/query/adapter; caption Việt giải thích vai trò.|[PNG14](../../presentation/implementation_report/qa_integrated/final/slide-14.png)|PASS|
|15|CLR, domainpositive và deltafit-only khớpcode; schematic rõ.|[PNG15](../../presentation/implementation_report/qa_integrated/final/slide-15.png)|PASS|
|16|Median-scale/aggregation đúngcode, anchors gốc không đổi; caption Việt.|[PNG16](../../presentation/implementation_report/qa_integrated/final/slide-16.png)|PASS|
|17|Giới hạn intercept-only gắn riêngsynthetic, không mặc địnhCOVID.|[PNG17](../../presentation/implementation_report/qa_integrated/final/slide-17.png)|PASS|
|18|ALR/reference/OLS rõ, captionViệt; W là shift tỷ lệ.|[PNG18](../../presentation/implementation_report/qa_integrated/final/slide-18.png)|PASS|
|19|D_c=D+1; chỉsyntheticD_c13; allocation/adaptercountlock phânbiệt.|[PNG19](../../presentation/implementation_report/qa_integrated/final/slide-19.png)|PASS|
|20|Khôi phục caveatCOVID5/5foldschạm50; stop/delta không đồngnhấthộitụ.|[PNG20](../../presentation/implementation_report/qa_integrated/final/slide-20.png)|PASS|
|21|Hai input donor/query tách, proposal quaadapter trướcfinalcounts.|[PNG21](../../presentation/implementation_report/qa_integrated/final/slide-21.png)|PASS|
|22|D_code2×JSD chuẩn, logtựnhiên, khôngsqrt; closureanchors.|[PNG22](../../presentation/implementation_report/qa_integrated/final/slide-22.png)|PASS|
|23|Alpha powers row-normalized, geometricpositive/supportcondition, côngthức khôngtràn.|[PNG23](../../presentation/implementation_report/qa_integrated/final/slide-23.png)|PASS|
|24|Residualcore, missingweights và adaptercounts tách; zero policies khai báo.|[PNG24](../../presentation/implementation_report/qa_integrated/final/slide-24.png)|PASS|
|25|PooledMAEpp,117675targets,5folds3seeds150locations, CI fixedfits hiện trênbody.|[PNG25](../../presentation/implementation_report/qa_integrated/final/slide-25.png)|PASS|
|26|4scenario panels, địnhnghĩacáchche, metricpp/CI và legend chung.|[PNG26](../../presentation/implementation_report/qa_integrated/final/slide-26.png)|PASS|
|27|Mốc0 nétđứt, pairedmacro/chieudau, fixedfits và multiplicity rõ.|[PNG27](../../presentation/implementation_report/qa_integrated/final/slide-27.png)|PASS|
|28|2truth×2masks, đơnvịpp,1seed/profile,noCI hiện trênbody.|[PNG28](../../presentation/implementation_report/qa_integrated/final/slide-28.png)|PASS|
|29|HardMCARsubset, đúngn637/576/467 và seed2026, không suy rộngranking.|[PNG29](../../presentation/implementation_report/qa_integrated/final/slide-29.png)|PASS|
|30|3kếtluận,1seed/nativemissing/fullpaperlimits vànextsteps.|[PNG30](../../presentation/implementation_report/qa_integrated/final/slide-30.png)|PASS|
|31|4referenceIEEE-style, tácgiả/DOI kiểm nguồn gốc; không bịa volume/pagesTsagris.|[PNG31](../../presentation/implementation_report/qa_integrated/final/slide-31.png)|PASS|
|32|Bảng6nhómfile-vaitrò, cỡchữ giữ nguyên; rút mô tả/đổi cột để không cắt cuối trang.|[PNG32](../../presentation/implementation_report/qa_integrated/final/slide-32.png)|PASS|

## Bằng chứng và lệnh

- [PDF cuối](../../presentation/implementation_report/implementation_report_final.pdf)
- [Contactsheet32trang](../../presentation/implementation_report/qa_integrated/final/contact-sheet-01-32.png)
- [Companionfile-role/source/terminology](INTEGRATED_FINAL_COMPANION_VI.md)
- [Source](../../presentation/implementation_report/main.tex)
- Harness `.harness/integrated-deck-32-2026-10-23/`: source.diff, numerical_audit.json, pdf_link_audit.json, final_artifacts.json, final_source_hashes.json, checks.json/checks.md/progress.md.

Buildtại presentation/implementation_report: `xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex`2lần. Lượtcuối: integrated-release-pass1.log và integrated-release-pass2.log, exit0,32pages, khôngwarning/error/overfull. Built-inCodexcompiler cũngsuccess. Render/verificationcommands trongcompanion.

## Phạm vi và giới hạn

PASS toàn bộ yêu cầu tích hợp. FAIL0, UNRESOLVED0. GiữHannover16:9,Arial,logo,palette3baseline; giữthứtự30mainframes. Dữliệu/scientificmetrics/algorithmkhôngđổi; sourcefootersthànhcitations, body/layoutđượcchỉnhtheofindings. LegacysectionsvàPDFticketcũkhôngxóa; bảncuốilàimplementation_report_final.pdf32trang. Khôngbenchmark/generate/retune/repeatedseeds. Scientificlimits(1seed/profile,nativemissingkhôngnhãn,fixedfitCI,chưahộitụ/fullpaper)làgiớihạnbáocáođãghi,khôngđượcđổiđểđạtQA.

## File thay đổi

- `presentation\implementation_report\main.tex`
- `presentation\implementation_report\sections\00_cover.tex`
- `presentation\implementation_report\sections\01_problem.tex`
- `presentation\implementation_report\sections\03_architecture.tex`
- `presentation\implementation_report\sections\04_runtime.tex`
- `presentation\implementation_report\sections\05_boundary.tex`
- `presentation\implementation_report\sections\06_ownership.tex`
- `presentation\implementation_report\sections\clarity_v2_07_capabilities.tex`
- `presentation\implementation_report\sections\23_synthetic.tex`
- `presentation\implementation_report\sections\ticket_01_frame09_generator.tex`
- `presentation\implementation_report\sections\ticket_01_frame10_profiles.tex`
- `presentation\implementation_report\sections\ticket02_11_missingness.tex`
- `presentation\implementation_report\sections\ticket02_12_synthetic_protocol.tex`
- `presentation\implementation_report\sections\ticket02_13_metrics.tex`
- `presentation\implementation_report\sections\ticket02_14_hron_module.tex`
- `presentation\implementation_report\sections\ticket02_15_hron_clr.tex`
- `presentation\implementation_report\sections\ticket02_16_hron_prediction.tex`
- `presentation\implementation_report\sections\ticket02_17_micoda_module.tex`
- `presentation\implementation_report\sections\ticket02_18_micoda_alr.tex`
- `presentation\implementation_report\sections\ticket02_19_micoda_allocation.tex`
- `presentation\implementation_report\sections\ticket02_20_micoda_loop.tex`
- `presentation\implementation_report\sections\ticket03_21_module.tex`
- `presentation\implementation_report\sections\ticket03_22_jsd.tex`
- `presentation\implementation_report\sections\ticket03_23_mean.tex`
- `presentation\implementation_report\sections\ticket03_24_allocate.tex`
- `presentation\implementation_report\sections\ticket03_25_primary.tex`
- `presentation\implementation_report\sections\ticket03_26_scenarios.tex`
- `presentation\implementation_report\sections\ticket03_27_paired.tex`
- `presentation\implementation_report\sections\ticket03_28_profiles.tex`
- `presentation\implementation_report\sections\ticket03_29_strata.tex`
- `presentation\implementation_report\sections\ticket03_30_conclusion.tex`
- `presentation\implementation_report\data\figure_manifest.json`
- `presentation\implementation_report\data\ticket03_figure_manifest.json`
- `presentation\implementation_report\figures\paired_plot.tex`
- Thêm `sections/integrated_31_references.tex`, `sections/integrated_32_file_roles.tex`.
- Sửa plotting builder `scripts/build_figures.py` để đườngmốc0 táixuấtđược.
- Thêm hai companion/report tích hợp, render/PDFvàharness.
