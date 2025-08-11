# Integrating ML Models (TensorFlow.js, ML Kit)

Approaches
- JS runtime: TensorFlow.js with tfjs-react-native (portable, slower)
- Native: ML Kit (Android/iOS), TFLite via native modules (faster)

TensorFlow.js (RN)
Install
- npm i @tensorflow/tfjs @tensorflow/tfjs-react-native
- Expo: npx expo install expo-gl react-native-unimodules (or the equivalent Expo modules in your SDK)

Example (simplified)
```ts
import '@tensorflow/tfjs-react-native';
import * as tf from '@tensorflow/tfjs';

await tf.ready();
const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
// preprocess image tensor and run inference
```

ML Kit (native)
- Android dependencies: com.google.mlkit:barcode-scanning, text-recognition, etc.
- iOS: via CocoaPods
- Create RN native module to pass images (YUV/RGB) to ML Kit and return results

Performance tips
- Prefer native inference for real-time camera use
- Run heavy work off the JS thread; use JSI/TurboModules for zero-bridge copies when possible
- Batch operations; reuse models; avoid reloading per frame

Privacy
- Process on-device when possible; redact PII; get explicit consent for uploads
