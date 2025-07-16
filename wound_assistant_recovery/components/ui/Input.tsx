import { Colors } from "@/constants/Colors";
import React from "react";
import {
    StyleSheet,
    Text,
    TextInput,
    TextStyle,
    View,
    ViewStyle,
} from "react-native";
import { SharedStyles } from "../../constants/SharedStyles";
import { IconSymbol } from "./IconSymbol";

interface InputProps {
  label?: string;
  icon?: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  secureTextEntry?: boolean;
  keyboardType?: "default" | "email-address" | "numeric" | "phone-pad";
  style?: ViewStyle;
  inputStyle?: TextStyle;
  autoCapitalize?: "none" | "sentences" | "words" | "characters";
  multiline?: boolean;
  numberOfLines?: number;
  error?: string;
}

const Input: React.FC<InputProps> = ({
  label,
  icon,
  value,
  onChangeText,
  placeholder,
  secureTextEntry = false,
  keyboardType = "default",
  style,
  inputStyle,
  autoCapitalize = "none",
  multiline = false,
  numberOfLines = 1,
  error,
}) => {
  return (
    <View style={[SharedStyles.formGroup, style]}>
      {label && <Text style={SharedStyles.label}>{label}</Text>}
      <View style={[styles.inputWrapper, error && styles.inputWrapperError]}>
        {icon && (
          <View style={styles.icon}>
            <IconSymbol
              name={icon as any}
              size={20}
              color={error ? Colors.light.danger : Colors.light.gray[400]}
            />
          </View>
        )}
        <TextInput
          style={[styles.input, inputStyle]}
          value={value}
          onChangeText={onChangeText}
          placeholder={placeholder}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          placeholderTextColor={Colors.light.gray[400]}
          multiline={multiline}
          numberOfLines={numberOfLines}
        />
      </View>
    </View>
  );
};

export default React.memo(Input);

const styles = StyleSheet.create({
  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.light.gray[50],
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: Colors.light.border,
  },
  inputWrapperError: {
    borderColor: Colors.light.danger,
    backgroundColor: "#fdf2f2",
  },
  icon: {
    marginRight: 12,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: Colors.light.text,
    minHeight: 24,
  },
});
