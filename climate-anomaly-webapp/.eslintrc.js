module.exports = {
  extends: ['react-app', 'react-app/jest'],
  rules: {
    // Reduce unused variable warnings to info level
    'no-unused-vars': 'off',
    // Reduce anchor accessibility warnings
    'jsx-a11y/anchor-is-valid': 'off',
    // Reduce script URL warnings
    'no-script-url': 'off',
    // Reduce React hooks dependency warnings
    'react-hooks/exhaustive-deps': 'off'
  }
};