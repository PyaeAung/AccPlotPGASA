# AccPlotPGASA

#ငလျင် Spectra Tools

ဒီ Repo ထဲမှာ Python နဲ့ ObsPy ကို အသုံးပြုပြီး  
**မြေငလျင် acceleration data (MiniSEED)** ကို  
- စက် response ဖယ်ရှား (Remove Instrument Response)  
- Acceleration ကို g unit အဖြစ် ပြောင်း  
- **5% damping Spectral Acceleration (Sa)** တွက်ချက်  
- Waveform နဲ့ Response Spectra ပုံဖော်  
လုပ်ဆောင်နိုင်မယ့် script များ ထည့်ထားပါတယ်။  

---

- MiniSEED + StationXML 
- Counts → Acceleration (m/s² သို့မဟုတ် g)  
- Newmark-β နည်းဖြင့် 5% damping Sa 
- Sa vs Period (0–1 s) + Waveform  
- **ASCE 7-16 Design Spectrum** နဲ့ overlay (5% damping)  

