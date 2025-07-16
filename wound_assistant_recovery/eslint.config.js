// https://docs.expo.dev/guides/using-eslint/
const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

// remove the errror or import React from 'react'


module.exports = defineConfig([
  expoConfig,
  {
    ignores: ['dist/*'],
  },
  
]);
