import React from "react";
import { View, ViewProps } from "react-native";
import { SharedStyles } from "../../constants/SharedStyles";

interface CardProps extends ViewProps {
  children: React.ReactNode;
  style?: any;
}

const Card: React.FC<CardProps> = ({ children, style, ...props }) => {
  return (
    <View style={[SharedStyles.card, style]} {...props}>
      {children}
    </View>
  );
};

export default Card;
