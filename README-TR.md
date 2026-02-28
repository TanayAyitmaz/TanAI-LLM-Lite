# TanAILite

## TanaAI ve TanAI-Lite nedir?
**TanAI-Lite, TanAI mimarisinin basitleştirilmiş açık kaynaklı sürümüdür. Gerçek model yapısının (GAT - Generative Adaptive Transformer), GPT sürümü halinde basitleştirilmiş / sadeleştirilmiş versiyonudur.**<br><br>
TanAI çekirdeğinde birçok modern yapı kullanır ve Transformer çekirdeğinde Fused, Ecv ve Chronos projeksiyonlarına sahiptir. *(Lite modelde bu mimariler yoktur)*
- **Fused**: Konu ve anlamsal bağlam tutarlılığı için 256D vektör projeksiyonu.
- **Ecv** *(emotional conditioning vector)*: Duygusal bağlam tutarlılığı için 64D vektör projeksiyonu. Ecv için Robert Plutchik'in 8 duygu yapısı kullanıldı ve binlerce duygusal cümle vektörlere dönüştürüldü. Bu veri kümesinden 48D duygu vektörü oluşturulur ve Duygusal Kullanıcı Profilinden 16D vektör oluşturulur..
- **Chronos**: Zaman frekanslarını öğrenmek için 32D vektör projeksiyonu. LLM'ler zaman serilerini bilmezler ve promptlardan net öğrenim sağlayamazlar. Chronos, geçmişi algılamak ve geleceği tahmin etmek için tasarlanmıştır.

SwiGLU, TanAI'da aktivasyon fonksiyonu olarak kullanıldı. GeLU, Tanai-Lite için kullanıldı.<br> 
AdaRMSNorm ve ada_proj, TanAI'da normalizasyon için kullanıldı. RMSNorm, Tanai-Lite için kullanıldı.<br> 
TanAI, monolitik olmayan modüler bir LLM'dir ve tüm kararlar **GlassBox - Telemetry** ile gözlemlenebilir.<br>
TanAI-Lite, birçok ekstra özelliğin basitleştirildiği ve açık kaynak olarak yayınlanan bir sürümdür.<br>

## TanAI-Lite Yapılandırması
TanAI'den esinlenerek geliştirilmiş minimal açık kaynaklı eğitim ve çıkarım yığını.

Model:<br> 
**TanAILiteGPT**(<br>
&emsp;(tok_emb): Embedding(32000, 512)<br>
&emsp;(pos_emb): Embedding(1024, 512)<br>
&emsp;(blocks): ModuleList(<br>
&emsp;&emsp;(0-7): 8 x TanAILiteBlock(<br>
&emsp;&emsp;&emsp;(norm_attn): TanAIRMSNorm()<br>
&emsp;&emsp;&emsp;(q_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(k_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(v_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(o_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(norm_mlp): TanAIRMSNorm()<br>
&emsp;&emsp;&emsp;(ff_up): Linear(in_features=512, out_features=2048, bias=False)<br>
&emsp;&emsp;&emsp;(ff_down): Linear(in_features=2048, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(dropout): Dropout(p=0.0, inplace=False)))<br>
&emsp;(norm): TanAIRMSNorm()<br>
&emsp;(lm_head): Linear(in_features=512, out_features=32000, bias=False))<br>

## Model parametre yapısı *(tied)*
Model Params: 42.08 M

## Donanım Gereksinimi *(30M-50M)*
TanAI-Lite Modeli ~42.082.816M parametreye sahiptir. Bu parametre yapısı TanAILiteConfig ile değiştirilebilir. Mevcut testlerde 16GB VRAM ile eğitim sorunsuz bir şekilde gerçekleştirilmiştir. Daha düşük eğitim parametreleri (düşük Batch-Size) kullanılarak 12GB VRAM ile eğitim gerçekleştirilebilir.
Önerilen:
- **3090 24GB** *(iyi eğitim çıktısı)*
- **4090 24GB** *(iyi eğitim çıktısı)*
- **5070TI 16GB** *(standard)*
- **5080 16GB** *(standard)*
- **5090 32GB** *(çok iyi eğitim çıktısı)*

Evet, bu yapılandırma RTX 3090/4090/5090 sınıfı kartlarla tek GPU üzerinde eğitilebilir.
- 24GB VRAM *(3090/4090)*: comfortable for 30M-50M with AdamW + mixed precision.
- 16GB VRAM *(some 50xx SKUs)*: still workable with lower batch + grad accumulation.
- Main pressure is activation memory (sequence length), not raw parameter count.

Practical guidance:
- `seq_len=1024`: basit
- `seq_len=2048`: pratik
- `seq_len=4096`: mümkün ancak daha küçük mikro parti gerektirir

## Scope
- Full open-source tokenizer / encoder / transformer
- Tokenizer training/eval
- Encoder training/eval
- Lite GPT model training
- SFT training
- Single-command inference

## Core CLIs
- `tanailite-corpus-slicer`
HF aracılığıyla istediğiniz dilde Corpus'u indirin ve test için Corpus'u bölebilirsiniz.
- `tanailite-train-tokenizer`
Corpus veri setiyle 32k vocab ile tokenizer'ı eğitebilirsiniz.
- `tanailite-train-encoder`
Encoder'ı tokenizer ve corpus veri setinizle eğitin. *(RAG ve Embedding vektör üretimi için)*
- `tanailite-train-base`
Modelinizi tamamen eğitebilirsiniz. *(Temel model, test için 5k adım için eğitilmiştir; bu eğitimi uzatarak çok daha iyi sonuçlar elde edebilirsiniz.)*
- `tanailite-train-sft`
Modelinize kişilik kazandırın (talimat yapısı). HF aracılığıyla bir Instruction / SFT veri setini indirin ve modelinizi eğitin.
- `tanailite-infer`
Modelin çıkarım testlerini gerçekleştirin.

## Nasıl çalıştıracağım?
> [!TIP]
> Geliştirme ortamı Python 3.10 ve üzerinde olmalıdır.
> Çalıştırmak için **docs/04_run.md** dosyasındaki talimatları ve komutları takip edebilirsiniz.

## Base Model ve Encoder Dosyaları nerede?
> [!IMPORTANT]
> **Base Model dosyası**: https://tanai.xyz/tanai/base_best.pt<br>
> Temel model sadece 5000 adımda eğitilmiştir ve henüz dili öğrenmemiştir. Lütfen en az 80-100 bin adımda eğitin ve çıkarım kontrollerini gerçekleştirin.
> 
> **Encoder dosyası**: https://tanai.xyz/tanai/encoder_best.pt<br>
> Encoder raporlarında retrieval_at1 > 0,7, mrr > 0,50, mean_margin > 0,05 gibi değerleri aşan enkoder çıkışlarının kullanılmasını öneririz. *(Bu enkoder dosyası, test için 300 adım boyunca eğitilmiştir.)*

## Raporlar
> [!NOTE]
> Eğitim raporları için **data/reports** klasöründeki JSON dosyalarını inceleyebilirsiniz.

## Dokümanlar
- 00_scope.md
- 01_architecture.md
- 02_training_flow.md
- 03_inference_flow.md
- 04_run.md
- 05_tanai_lite_info.md
- 06_corpus_selection.md
