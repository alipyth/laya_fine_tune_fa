# فاین‌تیون Laya برای فارسی — راهنمای قدم‌به‌قدم

[English](README.md)

این پوشه مدل `laya-multilingual` رو برای **دسته‌بندی پیام‌های پشتیبانی فارسی** آموزش می‌ده. بعد با بنچمارک خودت (`persian_fa`) نتیجه رو قبل و بعد مقایسه می‌کنه و آخرش مدل رو روی Hugging Face و کد رو روی GitHub می‌ذاره.

**سیستم تو:** GTX 1650 با ۴ گیگ حافظه‌ی گرافیک و ۱۶ گیگ رم. برای حالت پیش‌فرض (`--mode top`) کافیه. آموزش روی این کارت احتمالاً چند دقیقه تا نیم ساعت طول می‌کشه. این عددها تخمینی‌ان، چون من نتونستم روی کارت واقعی تستش کنم.

---

## ۰. ساختار پوشه‌ها

زیپ رو توی Downloads با «Extract All» باز کن. مثل دفعه‌ی قبل، ویندوز یه پوشه‌ی تو در تو می‌سازه. پس مسیرهایی که توی این راهنما استفاده می‌شن این‌ها هستن:
```
C:\Users\mr.j\Downloads\laya_fa_finetune\laya_fa_finetune      ← همین پروژه (پوشه‌ای که train.py توشه)
C:\Users\mr.j\Downloads\persian_fa\persian_fa      ← بنچمارک (از قبل داری)
```
اگه جای دیگه‌ای باز کردی، هر جا این دو مسیر اومده، مسیر خودت رو بذار.

## ۱. نصب PyTorch با پشتیبانی کارت گرافیک (مهم)

PyTorch معمولی که روی ویندوز نصب می‌شه **فقط با CPU کار می‌کنه**. اول چک کن:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
اگه `False` چاپ کرد:
1. درایور کارت گرافیک NVIDIA رو به‌روز کن. بعد دستور `nvidia-smi` رو بزن؛ بالا سمت راست «CUDA Version» رو نشون می‌ده.
2. PyTorch قبلی رو حذف کن:
   ```
   python -m pip uninstall -y torch
   ```
3. برو به https://pytorch.org/get-started/locally و این گزینه‌ها رو انتخاب کن: **Stable، Windows، Pip، Python** و یه نسخه‌ی **CUDA** که از عدد `nvidia-smi` بزرگ‌تر نباشه. دستوری که سایت می‌ده رو اجرا کن. شبیه این‌ـه:
   ```
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
4. دوباره چک کن. این بار باید `True` ببینی.

بعد پیش‌نیازهای بقیه رو نصب کن:
```
cd C:\Users\mr.j\Downloads\laya_fa_finetune\laya_fa_finetune
python -m pip install -r requirements.txt
```

## ۲. تست سریع (حدود یک دقیقه، بدون دانلود)
```
python tests\smoke_test.py
```
این تست با یه مدل کوچیکِ تصادفی کل مراحل رو امتحان می‌کنه: آموزش، ذخیره، بارگذاری دوباره و ساختن کارت مدل. باید `smoke test OK` ببینی. عدد دقتی که نشون می‌ده مهم نیست.

## ۳. بررسی داده
```
python check_data.py
```
داده‌ی آموزش ۲۰۰ پیامه (۵۰ تا برای هر برچسب) و داده‌ی dev ۴۰ پیام. هر دو توی پوشه‌ی `data\` هستن. این اسکریپت چک می‌کنه هیچ پیام آموزشی شبیه پیام‌های بنچمارک نباشه.

**خوندن پیام‌ها رو فراموش نکن.** این پیام‌ها رو هم من نوشتم، پس مثل بنچمارک بخونشون و اشتباه‌ها رو درست کن. اگه پیام تازه اضافه کردی، دوباره `check_data.py` رو بزن.

> ⚠️ فایل `heldout_benchmark_DO_NOT_TRAIN.jsonl` همون ۶۴ پیام بنچمارکه. فقط برای بررسی شباهت اینجاست. **هیچ‌وقت ازش برای آموزش استفاده نکن.**

## ۴. آموزش
```
python train.py --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
```
بار اول حدود ۷۰۰ مگ دانلود می‌شه (اگه قبلاً دانلود شده باشه، از همون استفاده می‌کنه). بعد این‌ها رو چاپ می‌کنه:
- دقت dev **قبل** از آموزش
- پیشرفت هر epoch
- ذخیره‌ی بهترین نسخه در `out\laya-fa-support\`

**اگه خطای کمبود حافظه (`CUDA out of memory`) گرفتی:**

| به ترتیب امتحان کن | دستور |
|---|---|
| batch کوچیک‌تر | `python train.py --batch 4 --revision ...` |
| + صرفه‌جویی در حافظه | `python train.py --batch 4 --grad-checkpointing --revision ...` |
| سبک‌ترین حالت | `python train.py --mode head --revision ...` |
| بدون کارت گرافیک (کندتر) | `python train.py --mode head --device cpu --revision ...` |

**حالت‌های آموزش:**
- `head`: فقط لایه‌های تصمیم‌گیری آموزش می‌بینن.
- `top` (پیش‌فرض): لایه‌های تصمیم‌گیری به‌علاوه‌ی ۴ لایه‌ی آخر مدل.
- `full`: همه‌ی لایه‌ها. برای ۴ گیگ حافظه زیاده.

## ۵. ارزیابی روی بنچمارک (مرحله‌ی اصلی)
```
cd C:\Users\mr.j\Downloads\persian_fa\persian_fa
python run.py --checkpoint C:\Users\mr.j\Downloads\laya_fa_finetune\laya_fa_finetune\out\laya-fa-support --output runs\finetuned
python audit.py --run-dir runs\finetuned --write
python compare.py results\v1\multilingual runs\finetuned --mode choice_fa
python compare.py results\v1\multilingual runs\finetuned --mode choice_en
```
حالا جدول قبل و بعد رو داری.

**قانون‌های صداقت. این‌ها مهم‌ترین بخش کارن:**
- **نتیجه رو هر چی شد گزارش کن.** ممکنه خیلی بهتر بشه، ممکنه کم بهتر بشه، و ممکنه حتی بدتر بشه. هر سه حالت نتیجه‌ی واقعی و قابل انتشارن.
- **با نگاه به نتیجه‌ی بنچمارک، داده‌ی آموزش رو عوض نکن.** اگه این کار رو بکنی، بنچمارک دیگه «تست مستقل» نیست. تنظیمات رو فقط با عدد dev عوض کن. حداکثر یکی دو بار روی بنچمارک ارزیابی کن و همه‌ی دفعه‌ها رو گزارش کن.
- **یه محدودیت رو صادقانه بگو:** داده‌ی آموزش و بنچمارک رو هر دو من با روش مشابه نوشتم. پس بهبود روی بنچمارک احتمالاً از بهبود روی پیام‌های واقعی مشتری بیشتره. این رو توی کارت مدل و README نوشتم.

## ۶. گذاشتن مدل روی Hugging Face
1. توی huggingface.co برو به **Settings → Access Tokens** و یه توکن با دسترسی **Write** بساز.
2. وارد شو:
   ```
   hf auth login
   ```
   توکن رو بچسبون. اگه پرسید «Add token as git credential?» جواب بده `n`.
3. اول فقط کارت مدل رو بساز و بخونش (هنوز چیزی آپلود نمی‌شه):
   ```
   cd C:\Users\mr.j\Downloads\laya_fa_finetune\laya_fa_finetune
   python push_to_hub.py --repo alipyth/laya-fa-support --before C:\Users\mr.j\Downloads\persian_fa\persian_fa\results\v1\multilingual\summary.json --after C:\Users\mr.j\Downloads\persian_fa\persian_fa\runs\finetuned\summary.json --dry-run
   ```
   فایل `out\laya-fa-support\README.md` رو باز کن و بخون.
4. اگه درست بود، همون دستور رو **بدون** `--dry-run` اجرا کن. حدود ۱.۳ گیگ آپلود می‌شه.
5. تست کن:
   ```
   python -c "import laya; print(laya.load('alipyth/laya-fa-support'))"
   ```

## ۷. گذاشتن کد روی GitHub
یه ریپوی جدید بساز با اسم `laya-persian-finetune`، بدون README. بعد:
```
cd C:\Users\mr.j\Downloads\laya_fa_finetune\laya_fa_finetune
git init
git add .
git commit -m "Persian support-triage fine-tuning for Laya"
git branch -M main
git remote add origin https://github.com/alipyth/laya-persian-finetune.git
git push -u origin main
```
پوشه‌ی `out\` (مدل ۱.۳ گیگی) خودکار کنار گذاشته می‌شه، چون جاش روی Hugging Face‌ـه، نه GitHub.

آخر کار، نتیجه‌ی قبل و بعد رو توی بخش Results فایل `README.md` همین ریپو بنویس. جدول‌های `compare.py` رو کپی کن.

## ۸. بعدش
- **ریپوی بنچمارک:** اجرای `runs\finetuned` رو در `results\v1\finetuned\` بذار و یه خط به README اضافه کن، با لینک مدل Hugging Face.
- **Issue:** توی Issue ریپوی Laya یه کامنت با نتیجه‌ی قبل و بعد بذار.
- **یوتیوب:** این نتیجه محتوای خوبی برای یه ویدیوئه.

## فایل‌ها

| فایل | کار |
|---|---|
| `data/train.jsonl`, `data/dev.jsonl` | ۲۰۰ پیام آموزش و ۴۰ پیام dev |
| `data/heldout_benchmark_DO_NOT_TRAIN.jsonl` | پیام‌های بنچمارک، فقط برای بررسی شباهت |
| `check_data.py` | بررسی برچسب‌ها، تکراری‌ها و شباهت به بنچمارک |
| `prompts_train.py` | ۴ پرامپت آموزشی (متفاوت با پرامپت‌های بنچمارک) |
| `augment.py` | تنوع نوشتاری (ی و ک عربی، نیم‌فاصله، ارقام) |
| `train.py` | آموزش، انتخاب بهترین epoch، کالیبراسیون و ذخیره |
| `push_to_hub.py` | ساختن کارت مدل و آپلود روی Hugging Face |
| `tests/smoke_test.py` | تست کل مسیر بدون دانلود |
| `tools/make_tiny_checkpoint.py` | مدل کوچیک تصادفی برای تست |
