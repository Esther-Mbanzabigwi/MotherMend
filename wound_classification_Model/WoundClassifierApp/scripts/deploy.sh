#!/bin/bash

echo "Starting deployment process..."

# Check for required tools
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required but not installed. Aborting." >&2; exit 1; }

# Install dependencies
echo "Installing dependencies..."
npm install

# Copy PyTorch model
echo "Copying PyTorch model..."
mkdir -p android/app/src/main/assets
cp ../wound_classifier_best.pth android/app/src/main/assets/

# Build Android Release
echo "Building Android release..."
cd android
./gradlew clean
./gradlew assembleRelease

# Check if build was successful
if [ -f "app/build/outputs/apk/release/app-release.apk" ]; then
    echo "Build successful! APK location: app/build/outputs/apk/release/app-release.apk"
else
    echo "Build failed!"
    exit 1
fi

# Instructions for next steps
echo """
Deployment completed!

To test the release build:
1. Install the APK on a device:
   adb install app/build/outputs/apk/release/app-release.apk

To publish to Play Store:
1. Create a Play Store developer account if you haven't
2. Create a new app in the Play Console
3. Upload the APK under Production track
4. Fill in store listing details
5. Submit for review

For iOS deployment:
1. Open ios/WoundClassifierApp.xcworkspace in Xcode
2. Select your team and bundle identifier
3. Build and archive the application
4. Submit to App Store through Xcode
""" 