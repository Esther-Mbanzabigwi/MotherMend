import React from 'react';
import {
    Image,
    ScrollView,
    StyleSheet,
    View,
    ViewStyle,
} from 'react-native';
import { Colors } from '../constants/Colors';
import { SharedStyles } from '../constants/SharedStyles';

interface AppLayoutProps {
  children: React.ReactNode;
  showIcon?: boolean;
  contentStyle?: ViewStyle;
  scrollEnabled?: boolean;
  useScrollView?: boolean;
}

export default function AppLayout({
  children,
  showIcon = true,
  contentStyle,
  scrollEnabled = true,
  useScrollView = true,
}: AppLayoutProps) {
  const Content = () => (
    <>
      {showIcon && (
        <View style={styles.iconContainer}>
          <Image
            source={require('../assets/images/baby_hand.png')}
            style={styles.iconImage}
            resizeMode="cover"
          />
        </View>
      )}
      {children}
    </>
  );

  return (
    <View style={styles.background}>
      {useScrollView ? (
        <ScrollView
          style={SharedStyles.container}
          contentContainerStyle={[styles.content, contentStyle]}
          showsVerticalScrollIndicator={false}
          scrollEnabled={scrollEnabled}
        >
          <Content />
        </ScrollView>
      ) : (
        <View style={[SharedStyles.container, styles.content, contentStyle]}>
          <Content />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    backgroundColor: Colors.light.background,
  },
  content: {
    padding: 20,
    paddingTop: 40,
    minHeight: '100%',
  },
  iconContainer: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: Colors.light.blue[50],
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.10,
    shadowRadius: 6,
    elevation: 3,
    overflow: 'hidden',
    borderColor: Colors.light.background,
    borderWidth: 2,
    alignSelf: 'center',
  },
  iconImage: {
    width: '100%',
    height: '100%',
    opacity: 0.95,
  },
}); 