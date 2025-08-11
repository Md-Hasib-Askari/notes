# Forms and Validation (react-hook-form + Yup)

Why react-hook-form
- Performance: uncontrolled inputs
- Clean API: Controller for custom inputs

Install
- npm i react-hook-form @hookform/resolvers yup

Basic form
```tsx
import { useForm, Controller } from 'react-hook-form';
import { View, TextInput, Text, Button } from 'react-native';
import * as yup from 'yup';
import { yupResolver } from '@hookform/resolvers/yup';

type Form = { amount: string; note?: string };

const schema = yup.object({
  amount: yup.string().required().matches(/^\d+(\.\d{1,2})?$/, 'Invalid number'),
  note: yup.string().max(100),
});

export default function TxForm({ onSubmit }: { onSubmit: (v: Form) => void }) {
  const { control, handleSubmit, formState: { errors, isSubmitting }, reset } = useForm<Form>({
    defaultValues: { amount: '', note: '' },
    resolver: yupResolver(schema),
    mode: 'onBlur',
  });

  return (
    <View style={{ gap: 12 }}>
      <Controller
        control={control}
        name="amount"
        render={({ field: { value, onChange, onBlur } }) => (
          <TextInput
            placeholder="Amount"
            keyboardType="decimal-pad"
            value={value}
            onChangeText={onChange}
            onBlur={onBlur}
            style={{ borderWidth: 1, borderColor: errors.amount ? 'red' : '#ccc', padding: 12, borderRadius: 8 }}
          />
        )}
      />
      {errors.amount && <Text style={{ color: 'red' }}>{errors.amount.message}</Text>}

      <Controller
        control={control}
        name="note"
        render={({ field: { value, onChange, onBlur } }) => (
          <TextInput
            placeholder="Note"
            value={value}
            onChangeText={onChange}
            onBlur={onBlur}
            style={{ borderWidth: 1, borderColor: '#ccc', padding: 12, borderRadius: 8 }}
          />
        )}
      />

      <Button title={isSubmitting ? 'Saving...' : 'Save'} onPress={handleSubmit((v) => { onSubmit(v); reset(); })} />
    </View>
  );
}
```

Tips
- Use setFocus to move between fields
- Combine with KeyboardAvoidingView
- For masks, use react-native-mask-text or similar
